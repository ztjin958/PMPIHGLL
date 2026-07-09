"""One-off: test set stats for piazza-pmidb sweep (drop=0)."""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(r"E:/JZT_XIAOLUNWEN")


def label_to_int(val) -> int:
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    if s in ("false", "0", "no"):
        return 0
    return int(float(s))


def read_pmi(path: Path) -> list[tuple[str, str, int]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        c0, c1 = r.fieldnames[0], r.fieldnames[1]
        lc = next(
            (n for n in ("target", "score", "label", "interaction") if n in r.fieldnames),
            r.fieldnames[2],
        )
        for row in r:
            m = row[c0].strip()
            p = row[c1].strip()
            if not m or not p:
                continue
            rows.append((m, p, label_to_int(row[lc])))
    return rows


def collect_nodes(features: Path) -> tuple[set[str], set[str]]:
    metas, proteins = set(), set()
    smi = features / "meta.smi"
    if smi.is_file():
        for line in open(smi, encoding="utf-8"):
            p = line.strip().rsplit(" ", 1)
            if len(p) == 2:
                metas.add(p[1].strip())
    for tsv in ("m_m_links.tsv", "p_p_links.tsv"):
        fp = features / tsv
        if not fp.is_file():
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    a, b = parts[0].strip(), parts[1].strip()
                    if tsv.startswith("m_m"):
                        metas.add(a)
                        metas.add(b)
                    else:
                        proteins.add(a)
                        proteins.add(b)
    seq = features / "matched_protein_sequences.csv"
    if seq.is_file():
        with open(seq, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            col = "Protein_ID" if "Protein_ID" in r.fieldnames else r.fieldnames[0]
            for row in r:
                proteins.add(row[col].strip())
    return metas, proteins


def main():
    train_p = REPO / "piazza" / "m_p_links.csv"
    test_p = REPO / "PMIDB" / "ecoil" / "m_p_links.csv"
    features = REPO / "piazza_pmidb_ecoil"

    train = read_pmi(train_p)
    test_raw = read_pmi(test_p)
    train_keys = {(m, p) for m, p, _ in train}
    train_metas = {m for m, _, _ in train}
    train_proteins = {p for _, p, _ in train}

    metas_g, proteins_g = collect_nodes(features)
    metas = metas_g | train_metas | {m for m, _, _ in test_raw}
    proteins = proteins_g | train_proteins | {p for _, p, _ in test_raw}

    test_rows: list[tuple[str, str, int]] = []
    overlap = 0
    skip_node = 0
    for m, p, y in test_raw:
        if (m, p) in train_keys:
            overlap += 1
            continue
        if m not in metas or p not in proteins:
            skip_node += 1
            continue
        test_rows.append((m, p, y))

    n_pos = sum(y for _, _, y in test_rows)
    n_neg = len(test_rows) - n_pos
    u_m = {m for m, _, _ in test_rows}
    u_p = {p for _, p, _ in test_rows}

    print("Mode: piazza-pmidb (sweep_drop_test_only_proteins.py)")
    print(f"  train PMI file: {train_p}")
    print(f"  test PMI file:  {test_p}")
    print(f"  drop_test_only_proteins=0, exclude train/test (m,p) overlap=True")
    print()
    print("=== Test set AFTER prepare (before rus_test) ===")
    print(f"  Metabolites (unique): {len(u_m)}")
    print(f"  Proteins (unique):    {len(u_p)}")
    print(f"  MPI total:            {len(test_rows)}")
    print(f"  MPI positive:         {n_pos}")
    print(f"  MPI negative:         {n_neg}")
    print(f"  (removed overlap with train: {overlap}, skipped missing nodes: {skip_node})")
    print()
    n_rus = 2 * min(n_pos, n_neg)
    print("=== Test set AFTER rus_test=True (sweep 实际评估) ===")
    print(f"  MPI total:            {n_rus}  (= 2*min(pos,neg))")
    print(f"  MPI positive:         {min(n_pos, n_neg)}")
    print(f"  MPI negative:         {min(n_pos, n_neg)}")
    print(f"  (与混淆矩阵规模对照: tn+fp+fn+tp 应 = {n_rus})")
    print()
    print("Note: unique meta/protein 在 RUS 后会变少（随机删负/正边），")
    print("      上表 unique 数为 RUS 前 prepare 输出的 test 边集合。")


if __name__ == "__main__":
    main()