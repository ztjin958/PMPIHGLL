"""
从 piazza/m_p_links.csv 筛选：
  - meta 出现在 stitch_ecoli 的代谢物节点集合中
  - protein 出现在 stitch_ecoli 的蛋白节点集合中
  - (meta, protein) 不在 stitch 的 m_p_links 中

分别对照 m_p_links_400.csv 与 m_p_links_700.csv，写出两个结果 CSV。
"""
import csv
from pathlib import Path

base = Path(r"E:/JZT_XIAOLUNWEN")
piazza_path = base / "piazza" / "m_p_links.csv"
stitch_400 = base / "stitch_ecoli" / "m_p_links_400.csv"
stitch_700 = base / "stitch_ecoli" / "m_p_links_700.csv"


def load_stitch(path):
    metas, proteins, pmis = set(), set(), set()
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m, p = row["meta"].strip(), row["protein"].strip()
            metas.add(m)
            proteins.add(p)
            pmis.add((m, p))
    return metas, proteins, pmis


def load_piazza(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        for row in r:
            rows.append(row)
    return fieldnames, rows


def filter_piazza(rows, metas, proteins, pmis):
    out = []
    for row in rows:
        m, p = row["meta"].strip(), row["protein"].strip()
        if m in metas and p in proteins and (m, p) not in pmis:
            out.append(row)
    return out


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    fieldnames, piazza_rows = load_piazza(piazza_path)
    p_metas = {r["meta"].strip() for r in piazza_rows}
    p_proteins = {r["protein"].strip() for r in piazza_rows}

    print("=" * 60)
    print("Piazza m_p_links.csv 统计")
    print("=" * 60)
    print(f"  总行数: {len(piazza_rows)}")
    print(f"  唯一代谢物(meta): {len(p_metas)}")
    print(f"  唯一蛋白(protein): {len(p_proteins)}")

    results = {}
    for label, stitch_path, out_name in [
        ("400", stitch_400, "m_p_links_piazza_in_stitch_nodes_not_in_pmi_400.csv"),
        ("700", stitch_700, "m_p_links_piazza_in_stitch_nodes_not_in_pmi_700.csv"),
    ]:
        metas, proteins, pmis = load_stitch(stitch_path)
        filtered = filter_piazza(piazza_rows, metas, proteins, pmis)
        out_path = base / "piazza" / out_name
        write_csv(out_path, fieldnames, filtered)
        results[label] = filtered

        meta_ov = len(p_metas & metas)
        pro_ov = len(p_proteins & proteins)
        # 仅蛋白在 stitch、且 PMI 不在 stitch（代谢物 CID 无交集时这是可达到的最大子集）
        protein_only = [
            r
            for r in piazza_rows
            if r["protein"].strip() in proteins and (r["meta"].strip(), r["protein"].strip()) not in pmis
        ]

        print(f"\n--- 对照 stitch_ecoli/m_p_links_{label}.csv ---")
        print(f"  STITCH 唯一 meta: {len(metas)}, 唯一 protein: {len(proteins)}, PMI 对数: {len(pmis)}")
        print(f"  Piazza 与 STITCH meta 交集: {meta_ov}")
        print(f"  Piazza 与 STITCH protein 交集: {pro_ov}")
        print(f"  满足 meta in STITCH 且 protein in STITCH 且 PMI not in STITCH: {len(filtered)}")
        print(f"  (参考) 仅 protein in STITCH 且 PMI not in STITCH: {len(protein_only)}")
        print(f"  已写入: {out_path}")

    k400 = {(r["meta"].strip(), r["protein"].strip()) for r in results["400"]}
    k700 = {(r["meta"].strip(), r["protein"].strip()) for r in results["700"]}
    print(f"\n两结果 PMI 交集: {len(k400 & k700)}（400/700 节点集相同则一致）")


if __name__ == "__main__":
    main()