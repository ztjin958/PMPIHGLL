"""
本地（节点级）5-fold × R 轮 RUS（默认 R=10，与 main.py 一致；PMIDB 入口脚本为 R=1），与 main.py / main_core 相同训练超参。

划分方式（local）：
  - split_mode=protein：按蛋白 ID 5 折；测试边 = 测试折蛋白上的所有 PMI
  - split_mode=meta：按代谢物 ID 5 折；测试边 = 测试折代谢物上的所有 PMI

用法:
  python main_local_cv.py --dataset stitch_ecoli_400 --split protein
  python main_local_cv.py --dataset piazza --split meta
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

from Model import Model
from prepare_core import run_prepare
from utils import balance_tensor, process_data

device = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_DATASETS = (
    "stitch_ecoli_400",
    "stitch_ecoli_700",
    "stitch_yeast_400",
    "stitch_yeast_700",
    "piazza",
    "pmidb_human",
)


def _node_folds(node_ids: list, n_splits: int, seed: int) -> list[list[str]]:
    """蛋白等节点较多时：打乱后轮转分配到各折。"""
    ids = list(node_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, nid in enumerate(ids):
        folds[i % n_splits].append(nid)
    return folds


def _meta_folds(meta_ids: list, n_splits: int, seed: int) -> list[list[str]]:
    """
    代谢物 local CV：每一折测试集至少包含 1 个 META。
    - 若 META 数 M >= n_splits：前 n_splits 个各独占一折，其余轮询并入（每折仍 >=1）。
    - 若 M < n_splits（如 PMIDB 的 4 个）：前 M 折用满全部不同 META，
      剩余折从已打乱列表中有放回各取 1 个（第五折可重复）。
    """
    ids = list(meta_ids)
    if not ids:
        return [[] for _ in range(n_splits)]
    rng = np.random.RandomState(seed)
    perm = list(ids)
    rng.shuffle(perm)
    m = len(perm)
    folds: list[list[str]] = [[] for _ in range(n_splits)]

    if m >= n_splits:
        for f in range(n_splits):
            folds[f].append(perm[f])
        for j in range(n_splits, m):
            folds[j % n_splits].append(perm[j])
    else:
        for f in range(m):
            folds[f].append(perm[f])
        for f in range(m, n_splits):
            folds[f].append(perm[int(rng.randint(0, m))])
    return folds


def _edge_train_test_mask(
    x_meta: np.ndarray,
    x_protein: np.ndarray,
    meta_id_list: list,
    protein_id_list: list,
    test_meta_ids: set[str],
    test_protein_ids: set[str],
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x_meta)
    test_mask = np.zeros(n, dtype=bool)
    if split_mode == "protein":
        for i in range(n):
            pid = protein_id_list[int(x_protein[i])]
            if pid in test_protein_ids:
                test_mask[i] = True
    elif split_mode == "meta":
        for i in range(n):
            mid = meta_id_list[int(x_meta[i])]
            if mid in test_meta_ids:
                test_mask[i] = True
    else:
        raise ValueError(split_mode)
    train_mask = ~test_mask
    return train_mask, test_mask


def run_local_cv(
    dataset_key: str,
    split_mode: str,
    prep: dict | None = None,
    *,
    processed: bool = False,
    R: int = 10,
    n_splits: int = 5,
    fold_seed: int = 42,
):
    if split_mode not in ("protein", "meta"):
        raise ValueError("split_mode must be 'protein' or 'meta'")
    if prep is None:
        prep = run_prepare(dataset_key, processed=processed)

    cfg = prep["cfg"]
    model_root = f"{cfg['model_dir']}_local_{split_mode}"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    meta_features = prep["meta_large_model_features"]
    protein_features = prep["protein_large_model_features"]
    protein_dim = protein_features.shape[1]
    meta_dim = meta_features.shape[1]

    meta_id_list = prep["meta_id_list"]
    protein_id_list = prep["protein_id_list"]

    hyperedge_protein = prep["hyperedge_protein_dis_knn_from_sim_index"]
    hyperedge_meta = prep["hyperedge_meta_dis_knn_from_sim_index"]

    if split_mode == "protein":
        node_ids = protein_id_list
        node_folds = _node_folds(node_ids, n_splits, fold_seed)
    else:
        node_ids = meta_id_list
        node_folds = _meta_folds(node_ids, n_splits, fold_seed)
        for fi, nf in enumerate(node_folds):
            if not nf:
                raise ValueError(
                    f"meta fold {fi + 1} has no test META (ids={len(node_ids)})"
                )

    lr = 0.001
    weight_dacay = 0.01
    step_size = 30
    threshold_cur_epoch = 30
    threshold_i = 240

    accuracy_scores = []
    specificity_scores = []
    precision_score_scores = []
    recall_scores = []
    roc_auc_scores = []
    f1_scores = []
    mcc_scores = []
    aupr_scores = []

    print(
        f"[local_cv] dataset={dataset_key}, split={split_mode}, "
        f"nodes={len(node_ids)}, model_dir={model_root}"
    )

    for item in range(R):
        y_full, delete_index = balance_tensor(torch.tensor(prep["Y"]))
        x_protein = np.delete(np.array(prep["X_proteins"]), delete_index)
        x_meta = np.delete(np.array(prep["X_metas"]), delete_index)

        for fold in range(n_splits):
            test_node_set = set(node_folds[fold])
            if split_mode == "protein":
                train_mask, test_mask = _edge_train_test_mask(
                    x_meta,
                    x_protein,
                    meta_id_list,
                    protein_id_list,
                    set(),
                    test_node_set,
                    "protein",
                )
            else:
                train_mask, test_mask = _edge_train_test_mask(
                    x_meta,
                    x_protein,
                    meta_id_list,
                    protein_id_list,
                    test_node_set,
                    set(),
                    "meta",
                )

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                print(
                    f"RUS {item + 1} Fold {fold + 1} skip "
                    f"(train={len(train_idx)}, test={len(test_idx)})"
                )
                continue

            print(
                f"RUS {item + 1} Fold {fold + 1} "
                f"[local-{split_mode}] train_edges={len(train_idx)} "
                f"test_edges={len(test_idx)}"
            )

            train_protein_idx = x_protein[train_idx].flatten()
            train_meta_idx = x_meta[train_idx].flatten()
            test_protein_idx = x_protein[test_idx].flatten()
            test_meta_idx = x_meta[test_idx].flatten()
            y_train = torch.tensor(y_full[train_idx].numpy()).to(device)
            y_val = torch.tensor(y_full[test_idx].numpy()).to(device)

            protein_data = Data(
                x=torch.tensor(protein_features, dtype=torch.float32).to(device),
                hyperedge_index=hyperedge_protein,
            ).to(device)
            meta_data = Data(
                x=torch.tensor(meta_features, dtype=torch.float32).to(device),
                hyperedge_index=hyperedge_meta,
            ).to(device)
            data = {
                "protein_dis_knn_from_sim_data": protein_data,
                "meta_dis_knn_from_sim_data": meta_data,
            }

            model = Model(protein_dim, meta_dim).to(device)
            optimizer = torch.optim.Adam(
                params=model.parameters(), lr=lr, weight_decay=weight_dacay
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.5
            )
            criterion = nn.BCEWithLogitsLoss().to(device)

            fold_dir = os.path.join(
                model_root, f"rus_{item + 1}", f"fold_{fold + 1}"
            )
            if item == 0 and fold == 0 and os.path.exists(model_root):
                shutil.rmtree(model_root)
            os.makedirs(fold_dir, exist_ok=True)

            model.train()
            print("开始训练")
            best_loss = 1e5
            cur_epoch = 0
            ckpt = os.path.join(fold_dir, "best_model.pth")

            for i in tqdm(range(1000), file=sys.stdout):
                output, cl_loss = model(
                    data, index=(train_protein_idx, train_meta_idx)
                )
                loss = criterion(output.view(-1), y_train.float())
                optimizer.zero_grad()
                total_loss = loss + cl_loss
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if i and i % 50 == 0:
                    labels = y_train.float().to("cpu").flatten()
                    scores = nn.functional.sigmoid(
                        output.detach().flatten().to("cpu")
                    )
                    predicted = scores >= 0.5
                    mcc = matthews_corrcoef(labels, predicted)
                    roc_auc = roc_auc_score(labels, scores)
                    accuracy = accuracy_score(labels, predicted)
                    print(mcc, roc_auc, accuracy, total_loss.item())

                if best_loss > total_loss.item():
                    best_loss = total_loss.item()
                    cur_epoch = 0
                    torch.save(model.state_dict(), ckpt)
                else:
                    cur_epoch += 1
                    if cur_epoch > threshold_cur_epoch and i > threshold_i:
                        break

            model.eval()
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print("开始测试")

            with torch.no_grad():
                output, _ = model(
                    data, index=(test_protein_idx, test_meta_idx)
                )
                labels = y_val.float().to("cpu").flatten()
                scores = nn.functional.sigmoid(
                    output.detach().flatten().to("cpu")
                )
                predicted = scores >= 0.5
                mcc = matthews_corrcoef(labels, predicted)
                prec = precision_score(labels, predicted, zero_division=0)
                recall = recall_score(labels, predicted, zero_division=0)
                precision, recall_aupr, _ = precision_recall_curve(
                    labels, scores
                )
                aupr = auc(recall_aupr, precision)
                roc_auc = roc_auc_score(labels, scores)
                accuracy = accuracy_score(labels, predicted)
                f1 = f1_score(labels, predicted, average="binary", zero_division=0)
                tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()
                specificity = tn / (tn + fp) if (tn + fp) else 0.0
                print(
                    f"accuracy:{accuracy}",
                    f"f1:{f1}",
                    f"roc_auc:{roc_auc}",
                    f"aupr:{aupr}",
                    sep=" ",
                )
                print(
                    f"mcc:{mcc}",
                    f"specificity:{specificity}",
                    f"precision_score:{prec}",
                    sep=" ",
                )
                accuracy_scores.append(accuracy)
                specificity_scores.append(specificity)
                precision_score_scores.append(prec)
                recall_scores.append(recall)
                roc_auc_scores.append(roc_auc)
                f1_scores.append(f1)
                mcc_scores.append(mcc)
                aupr_scores.append(aupr)

    n_done = len(accuracy_scores)
    if n_done != R * n_splits:
        print(
            f"[warn] completed runs {n_done}, expected {R * n_splits}; "
            "summary uses actual count"
        )

    print("*********************************************")
    print(f"[local_cv] split_mode={split_mode}, runs={n_done}")
    if n_done >= n_splits:
        r_eff = n_done // n_splits
        print("accuracy_scores:", process_data(accuracy_scores, r_eff, n_splits))
        print("specificity_scores:", process_data(specificity_scores, r_eff, n_splits))
        print(
            "precision_score_scores:",
            process_data(precision_score_scores, r_eff, n_splits),
        )
        print("aupr_scores:", process_data(aupr_scores, r_eff, n_splits))
        print("recall_scores:", process_data(recall_scores, r_eff, n_splits))
        print("roc_auc_scores:", process_data(roc_auc_scores, r_eff, n_splits))
        print("f1_scores:", process_data(f1_scores, r_eff, n_splits))
        print("mcc_scores:", process_data(mcc_scores, r_eff, n_splits))
    return prep


def main():
    parser = argparse.ArgumentParser(description="Local 5-fold CV (protein or meta)")
    parser.add_argument(
        "--dataset",
        choices=LOCAL_DATASETS,
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("protein", "meta"),
        required=True,
        help="protein: 按蛋白节点划分; meta: 按代谢物节点划分",
    )
    parser.add_argument("--R", type=int, default=10, help="RUS 重复轮数（默认 10）")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument(
        "--processed",
        action="store_true",
        help="使用 *_processed.csv（默认使用原始 m_p_links*.csv）",
    )
    args = parser.parse_args()
    run_local_cv(
        args.dataset,
        args.split,
        processed=args.processed,
        R=args.R,
        n_splits=args.folds,
        fold_seed=args.fold_seed,
    )


if __name__ == "__main__":
    main()