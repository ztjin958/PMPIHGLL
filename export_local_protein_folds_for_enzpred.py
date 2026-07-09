"""
导出与 main_local_cv._node_folds 完全相同的蛋白 5 折，供 enz-pred kfold-seq 使用。

enz-pred 按 HTS 表中的 SEQ（氨基酸序列）匹配 test 折；本脚本把每折留出的蛋白 ID
映射为 sequence，写入 pickle（--split-groups-file）。

用法:
  python export_local_protein_folds_for_enzpred.py
  python export_local_protein_folds_for_enzpred.py --dataset stitch_ecoli_400 --fold-seed 42

生成:
  enz-pred/data/processed/split_groups/<dataset>_protein_local_folds_seq.p
  enz-pred/data/processed/split_groups/<dataset>_protein_local_folds.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle

import numpy as np

from datasets_processed import RAW_LINKS_FILE, REPO, get_dataset


def _node_folds(node_ids: list, n_splits: int, seed: int) -> list[list[str]]:
    """与 main_local_cv._node_folds 相同（避免 import torch）。"""
    ids = list(node_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for i, nid in enumerate(ids):
        folds[i % n_splits].append(nid)
    return folds


def load_protein_id_list_from_links(links_path: str) -> list[str]:
    """与 prepare_core 一致：PMI 表第二列去重顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    with open(links_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header: {links_path}")
        # meta, protein, target
        pro_col = "protein" if "protein" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            pid = str(row[pro_col]).strip()
            if pid and pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


def load_protein_id_to_seq(csv_path: str) -> dict[str, str]:
    m: dict[str, str] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header: {csv_path}")
        id_col = reader.fieldnames[0]
        seq_col = reader.fieldnames[1] if len(reader.fieldnames) > 1 else "Sequence"
        for row in reader:
            pid = str(row[id_col]).strip()
            seq = str(row[seq_col]).strip()
            if pid and seq:
                m[pid] = seq
    return m


def export_folds(
    dataset_key: str,
    *,
    processed: bool = False,
    n_splits: int = 5,
    fold_seed: int = 42,
    out_dir: str | None = None,
) -> str:
    cfg = get_dataset(dataset_key, processed=processed)
    features_dir = cfg["features_dir"]
    links_name = (
        cfg["links_file"]
        if processed
        else RAW_LINKS_FILE.get(dataset_key, cfg["links_file"])
    )
    links_path = os.path.join(features_dir, links_name)
    seq_csv = os.path.join(features_dir, "matched_protein_sequences.csv")

    protein_ids = load_protein_id_list_from_links(links_path)
    id_to_seq = load_protein_id_to_seq(seq_csv)
    folds = _node_folds(protein_ids, n_splits, fold_seed)

    split_groups: dict[str, list[str]] = {}
    fold_report = []
    missing_seq = []

    for fi, fold_pids in enumerate(folds):
        seqs = []
        for pid in fold_pids:
            seq = id_to_seq.get(pid)
            if not seq:
                missing_seq.append(pid)
                continue
            seqs.append(seq)
        key = f"Fold_{fi}"
        split_groups[key] = seqs
        fold_report.append(
            {
                "fold": fi,
                "n_proteins": len(fold_pids),
                "n_seqs": len(seqs),
                "protein_ids_sample": fold_pids[:5],
            }
        )

    if out_dir is None:
        out_dir = os.path.join(REPO, "enz-pred", "data", "processed", "split_groups")
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{dataset_key}_protein_local_folds"
    pkl_path = os.path.join(out_dir, f"{tag}_seq.p")
    json_path = os.path.join(out_dir, f"{tag}.json")

    with open(pkl_path, "wb") as f:
        pickle.dump(split_groups, f)

    meta = {
        "dataset_key": dataset_key,
        "processed_links": processed,
        "links_path": os.path.abspath(links_path),
        "n_splits": n_splits,
        "fold_seed": fold_seed,
        "algorithm": "main_local_cv._node_folds (same as main_local_stitch_ecoli_400_protein.py)",
        "enzpred_arg": f"--split-groups-file {pkl_path}",
        "enzpred_note": (
            "kfold-seq + split_groups: test = rows whose SEQ is in Fold_k. "
            "Remaining rows split train/val by val-size (local GNN uses 100% non-test for train)."
        ),
        "folds": fold_report,
        "missing_seq_protein_ids": missing_seq,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"\nWrote: {pkl_path}")
    return pkl_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="stitch_ecoli_400")
    p.add_argument(
        "--processed",
        action="store_true",
        help="用 *_processed.csv（默认与 main_local_stitch_ecoli_400_protein 一致为 False）",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=42)
    args = p.parse_args()
    export_folds(
        args.dataset,
        processed=args.processed,
        n_splits=args.folds,
        fold_seed=args.fold_seed,
    )


if __name__ == "__main__":
    main()