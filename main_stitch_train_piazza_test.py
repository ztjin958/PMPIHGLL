"""
传导式 train/test（单次，无交叉验证）：
构图使用合并目录全部节点；训练/测试 PMI 来自不同源。

模式:
  stitch-piazza（默认）: 训练 stitch_ecoli，测试 piazza，特征 stitch_piazza_ecoli_{400|700}
  piazza-pmidb: 训练 piazza，测试 PMIDB/ecoil，特征 piazza_pmidb_ecoil（先跑 merge_piazza_pmidb_ecoli.py）

用法:
  conda activate pyjzt
  python main_stitch_train_piazza_test.py --label 400
  python main_stitch_train_piazza_test.py --mode piazza-pmidb

  CMD 下若遇 OpenMP 冲突可先执行: set KMP_DUPLICATE_LIB_OK=TRUE
"""
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
from prepare_transductive_stitch_piazza import (
    default_prepare_400,
    default_prepare_700,
    default_prepare_piazza_train_pmidb_test,
)
from utils import balance_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"


def _apply_rus(meta_idx, protein_idx, y):
    y_bal, delete_index = balance_tensor(torch.tensor(y))
    if delete_index.numel() > 0:
        di = delete_index.numpy()
        meta_idx = np.delete(meta_idx, di)
        protein_idx = np.delete(protein_idx, di)
    y = y_bal.numpy().astype(np.int64)
    return meta_idx, protein_idx, y


def run_stitch_train_piazza_test(
    prep: dict,
    *,
    use_rus: bool = True,
    rus_test: bool = False,
    train_tag: str = "STITCH",
    test_tag: str = "Piazza",
    run_seed: int = 42,
    quiet: bool = False,
):
    cfg = prep["cfg"]
    model_root = cfg["model_dir"]

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    meta_features = prep["meta_large_model_features"]
    protein_features = prep["protein_large_model_features"]
    protein_dim = protein_features.shape[1]
    meta_dim = meta_features.shape[1]

    train_meta = prep["train_meta_idx"]
    train_protein = prep["train_protein_idx"]
    train_Y = prep["train_Y"]
    test_meta = prep["test_meta_idx"]
    test_protein = prep["test_protein_idx"]
    test_Y = prep["test_Y"]

    n_train_raw = len(train_Y)
    n_train_pos_raw = int(np.sum(train_Y))
    if use_rus:
        if not quiet:
            print(
                f"[main] RUS on train set ({train_tag}): "
                f"{n_train_raw} edges (pos={n_train_pos_raw}) -> 1:1 pos/neg"
            )
        train_meta, train_protein, train_Y = _apply_rus(
            train_meta, train_protein, train_Y
        )
    elif not quiet:
        print(
            f"[main] train set ({train_tag}): NO RUS, "
            f"{n_train_raw} edges (pos={n_train_pos_raw}, neg={n_train_raw - n_train_pos_raw})"
        )
    if rus_test:
        n_test_raw = len(test_Y)
        if not quiet:
            print(
                f"[main] RUS on test set ({test_tag}): "
                f"{n_test_raw} edges -> 1:1 pos/neg"
            )
        test_meta, test_protein, test_Y = _apply_rus(
            test_meta, test_protein, test_Y
        )

    train_protein_idx = train_protein.flatten()
    train_meta_idx = train_meta.flatten()
    test_protein_idx = test_protein.flatten()
    test_meta_idx = test_meta.flatten()
    y_train = torch.tensor(train_Y).to(device)
    y_val = torch.tensor(test_Y).to(device)

    hyperedge_protein = prep["hyperedge_protein_dis_knn_from_sim_index"]
    hyperedge_meta = prep["hyperedge_meta_dis_knn_from_sim_index"]

    lr = 0.001
    weight_dacay = 0.01
    step_size = 30
    threshold_cur_epoch = 30
    threshold_i = 240

    if not quiet:
        print(f"[main] {train_tag}_train / {test_tag}_test, model_dir={model_root}")
        print(f"[main] train samples={len(y_train)}, test samples={len(y_val)}")

    model = Model(protein_dim, meta_dim).to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_dacay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=0.5
    )
    criterion = nn.BCEWithLogitsLoss().to(device)

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

    model.train()
    if not quiet:
        print(f"开始训练（{train_tag} PMI）")
    if os.path.exists(model_root):
        shutil.rmtree(model_root)
    os.makedirs(model_root, exist_ok=True)
    ckpt = os.path.join(model_root, "best_model.pth")

    best_loss = 1e5
    cur_epoch = 0

    train_iter = range(1000)
    if quiet:
        train_iter = tqdm(train_iter, disable=True)
    else:
        train_iter = tqdm(train_iter, file=sys.stdout)
    for i in train_iter:
        output, cl_loss = model(data, index=(train_protein_idx, train_meta_idx))
        loss = criterion(output.view(-1), y_train.float())
        optimizer.zero_grad()
        total_loss = loss + cl_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if not quiet and i and i % 50 == 0:
            labels = y_train.float().to("cpu").flatten()
            scores = nn.functional.sigmoid(output.detach().flatten().to("cpu"))
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
    if not quiet:
        print(f"开始测试（{test_tag} PMI）")

    with torch.no_grad():
        output, _ = model(data, index=(test_protein_idx, test_meta_idx))
        labels = y_val.float().to("cpu").flatten()
        scores = nn.functional.sigmoid(output.detach().flatten().to("cpu"))
        predicted = scores >= 0.5
        mcc = matthews_corrcoef(labels, predicted)
        prec = precision_score(labels, predicted, zero_division=0)
        recall = recall_score(labels, predicted, zero_division=0)
        precision, recall_aupr, _ = precision_recall_curve(labels, scores)
        aupr = auc(recall_aupr, precision)
        roc_auc = roc_auc_score(labels, scores)
        accuracy = accuracy_score(labels, predicted)
        f1 = f1_score(labels, predicted, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

    if not quiet:
        print("*********************************************")
        print("confusion matrix (rows=true, cols=pred; 0=neg, 1=pos):")
        print(f"              pred_0    pred_1")
        print(f"  true_0 (TN)   {tn:6d}    {fp:6d}")
        print(f"  true_1 (FN)   {fn:6d}    {tp:6d}")
        print(f"  tn={tn}, fp={fp}, fn={fn}, tp={tp}")
        print(f"accuracy:{accuracy}", f"f1:{f1}", f"roc_auc:{roc_auc}", f"aupr:{aupr}")
        print(
            f"mcc:{mcc}",
            f"specificity:{specificity}",
            f"precision:{prec}",
            f"recall:{recall}",
        )
    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "aupr": aupr,
        "mcc": mcc,
        "specificity": specificity,
        "precision": prec,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("stitch-piazza", "piazza-pmidb"),
        default="stitch-piazza",
        help="stitch-piazza: STITCH 训练 / Piazza 测试；piazza-pmidb: Piazza 训练 / PMIDB/ecoil 测试",
    )
    parser.add_argument("--label", choices=("400", "700"), default="400")
    parser.add_argument(
        "--processed",
        action="store_true",
        help="使用 *_processed.csv 版 m_p_links（默认用原始 CSV）",
    )
    parser.add_argument(
        "--no-rus",
        action="store_true",
        help="训练集不做随机欠采样平衡",
    )
    parser.add_argument(
        "--rus-test",
        action="store_true",
        help="测试集也做 RUS，正负 1:1（对比实验用）",
    )
    parser.add_argument(
        "--drop-test-only-proteins",
        type=float,
        default=0.0,
        metavar="FRAC",
        help=(
            "在去掉与训练重复的测试边之后，随机剔除 FRAC 比例的"
            "「仅在测试边出现、训练 PMI 未出现」的蛋白：删除其测试边，"
            "并从构图节点 / p_p 超边 / 蛋白特征矩阵中移除（0~1，如 0.9）"
        ),
    )
    parser.add_argument(
        "--drop-test-only-meta",
        type=float,
        default=0.0,
        metavar="FRAC",
        help=(
            "同上，但针对代谢物（meta）：删除测试边并从 m_m 超边 / 代谢物特征矩阵中"
            "移除对应节点（0~1，如 0.9；随机种子 43，与蛋白侧 42 独立）"
        ),
    )
    args = parser.parse_args()
    processed = args.processed
    drop_frac = max(0.0, min(1.0, args.drop_test_only_proteins))
    drop_meta_frac = max(0.0, min(1.0, args.drop_test_only_meta))
    if args.mode == "piazza-pmidb":
        prep = default_prepare_piazza_train_pmidb_test(
            piazza_processed=processed,
            pmidb_processed=processed,
            drop_test_only_protein_fraction=drop_frac,
            drop_test_only_meta_fraction=drop_meta_frac,
        )
        run_stitch_train_piazza_test(
            prep,
            use_rus=not args.no_rus,
            rus_test=args.rus_test,
            train_tag="Piazza",
            test_tag="PMIDB/ecoil",
        )
    else:
        if args.label == "400":
            prep = default_prepare_400(
                stitch_processed=processed,
                piazza_processed=processed,
                drop_test_only_protein_fraction=drop_frac,
                drop_test_only_meta_fraction=drop_meta_frac,
            )
        else:
            prep = default_prepare_700(
                stitch_processed=processed,
                piazza_processed=processed,
                drop_test_only_protein_fraction=drop_frac,
                drop_test_only_meta_fraction=drop_meta_frac,
            )
        run_stitch_train_piazza_test(
            prep,
            use_rus=not args.no_rus,
            rus_test=args.rus_test,
            train_tag="STITCH",
            test_tag="Piazza",
        )


if __name__ == "__main__":
    main()