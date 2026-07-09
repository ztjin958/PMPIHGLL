"""
将 piazza/m_p_links.csv 与 PMIDB/ecoil/m_p_links.csv 合并，
并合并、去重相关辅助文件（meta.smi、matched_protein_sequences 等），
写入新目录且每条记录标注数据来源 source。

m_m_links / p_p_links：
  - 两侧均使用各自目录下已有的 m_m_links.tsv、p_p_links.tsv 融合，
    再按合并后 PMI 的代谢物/蛋白节点集合过滤（与 stitch_piazza 流程一致，但不从 STITCH 原始库重建）。

输出目录：
  piazza_pmidb_ecoil/
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

BASE = Path(r"E:/JZT_XIAOLUNWEN")
PIAZZA_DIR = BASE / "piazza"
PMIDB_DIR = BASE / "PMIDB" / "ecoil"

PIAZZA_M_P = PIAZZA_DIR / "m_p_links.csv"
PMIDB_M_P = PMIDB_DIR / "m_p_links.csv"

OUT_DIR = BASE / "piazza_pmidb_ecoil"

SOURCE_PIAZZA = "piazza"
SOURCE_PMIDB = "pmidb_ecoil"


def _is_positive(val: str) -> bool:
    v = str(val).strip()
    if v in ("1", "True", "true", "TRUE"):
        return True
    if v in ("0", "False", "false", "FALSE"):
        return False
    try:
        return float(v) == 1.0
    except ValueError:
        return False


def _label_str(positive: bool, sample: str) -> str:
    v = str(sample).strip()
    if v in ("True", "False", "true", "false", "TRUE", "FALSE"):
        return "True" if positive else "False"
    return "1" if positive else "0"


def read_m_p_csv(path: Path, source: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = (
                row.get("meta")
                or row.get("KEGG")
                or row.get("kegg")
                or ""
            ).strip()
            protein = (
                row.get("protein")
                or row.get("Uniprot_KB_id")
                or row.get("uniprot_kb_id")
                or ""
            ).strip()
            if not meta or not protein:
                continue
            if "target" in row:
                label = row["target"]
            elif "score" in row:
                label = row["score"]
            elif "interaction" in row:
                label = row["interaction"]
            else:
                label = "0"
            rows.append(
                {
                    "meta": meta,
                    "protein": protein,
                    "target": label,
                    "source": source,
                }
            )
    return rows


def merge_m_p(pmidb_path: Path, piazza_path: Path) -> tuple[list[dict], dict]:
    pmidb_rows = read_m_p_csv(pmidb_path, SOURCE_PMIDB)
    piazza_rows = read_m_p_csv(piazza_path, SOURCE_PIAZZA)

    by_key: dict[tuple[str, str], dict] = {}
    stats = {
        "pmidb_rows": len(pmidb_rows),
        "piazza_rows": len(piazza_rows),
        "duplicate_keys": 0,
        "merged_rows": 0,
    }

    def upsert(row: dict):
        key = (row["meta"], row["protein"])
        if key not in by_key:
            by_key[key] = dict(row)
            return
        stats["duplicate_keys"] += 1
        existing = by_key[key]
        src_a = existing["source"]
        src_b = row["source"]
        if src_a != src_b:
            parts = sorted(set(src_a.split(";") + src_b.split(";")))
            existing["source"] = ";".join(parts)
        pos = _is_positive(existing["target"]) or _is_positive(row["target"])
        existing["target"] = _label_str(pos, existing["target"])

    for r in pmidb_rows:
        upsert(r)
    for r in piazza_rows:
        upsert(r)

    merged = sorted(by_key.values(), key=lambda x: (x["meta"], x["protein"]))
    stats["merged_rows"] = len(merged)
    stats["only_pmidb"] = sum(1 for r in merged if r["source"] == SOURCE_PMIDB)
    stats["only_piazza"] = sum(1 for r in merged if r["source"] == SOURCE_PIAZZA)
    stats["both_sources"] = sum(1 for r in merged if ";" in r["source"])
    return merged, stats


def write_m_p(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["meta", "protein", "target", "source"])
        w.writeheader()
        w.writerows(rows)


def read_tsv_edges(path: Path, source: str) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            w = parts[2].strip() if len(parts) > 2 else ""
            rows.append({"node1": a, "node2": b, "weight": w, "source": source})
    return rows


def merge_undirected_edges(
    rows_a: list[dict], rows_b: list[dict]
) -> tuple[list[dict], dict]:
    by_key: dict[tuple[str, str], dict] = {}
    dup = 0

    def upsert(row: dict):
        nonlocal dup
        n1, n2 = row["node1"], row["node2"]
        key = (n1, n2) if n1 <= n2 else (n2, n1)
        if key not in by_key:
            by_key[key] = {
                "node1": key[0],
                "node2": key[1],
                "weight": row["weight"],
                "source": row["source"],
            }
            return
        dup += 1
        ex = by_key[key]
        parts = sorted(set(ex["source"].split(";") + row["source"].split(";")))
        ex["source"] = ";".join(parts)
        if row["weight"]:
            try:
                if float(row["weight"] or 0) >= float(ex["weight"] or 0):
                    ex["weight"] = row["weight"]
            except ValueError:
                pass

    for r in rows_a:
        upsert(r)
    for r in rows_b:
        upsert(r)

    merged = sorted(by_key.values(), key=lambda x: (x["node1"], x["node2"]))
    return merged, {"duplicate_keys": dup, "merged_rows": len(merged)}


def write_tsv_with_source(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            w = r["weight"]
            f.write(f"{r['node1']}\t{r['node2']}\t{w}\t{r['source']}\n")


def write_tsv_training(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            w = r["weight"] if r["weight"] != "" else "0"
            f.write(f"{r['node1']}\t{r['node2']}\t{w}\n")


def _parse_meta_smi_line(line: str) -> tuple[str, str] | None:
    """piazza: 'SMILES CID'（空格）；PMIDB/ecoil: 'SMILES\\tCID'（制表符）。"""
    line = line.strip()
    if not line:
        return None
    if "\t" in line:
        parts = line.split("\t")
        if len(parts) >= 2:
            return parts[0].strip(), parts[-1].strip()
        return None
    parts = line.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None


def merge_meta_smi(pmidb_path: Path, piazza_path: Path) -> tuple[list[dict], dict]:
    """按 CID 去重，优先保留 pmidb_ecoil 的 SMILES。"""
    by_cid: dict[str, tuple[str, str]] = {}
    order: list[str] = []

    def add_line(line: str, source: str):
        parsed = _parse_meta_smi_line(line)
        if parsed is None:
            return
        smi, cid = parsed
        if cid not in by_cid:
            by_cid[cid] = (smi, source)
            order.append(cid)
            return
        old_smi, old_src = by_cid[cid]
        if old_src == source:
            return
        parts_src = sorted(set(old_src.split(";") + source.split(";")))
        keep_smi = old_smi if SOURCE_PMIDB in old_src.split(";") else smi
        by_cid[cid] = (keep_smi, ";".join(parts_src))

    if pmidb_path.is_file():
        with open(pmidb_path, encoding="utf-8") as f:
            for line in f:
                add_line(line, SOURCE_PMIDB)
    if piazza_path.is_file():
        with open(piazza_path, encoding="utf-8") as f:
            for line in f:
                add_line(line, SOURCE_PIAZZA)

    records = []
    both = only_m = only_p = 0
    for cid in order:
        smi, src = by_cid[cid]
        records.append({"smi": smi, "cid": cid, "source": src})
        if ";" in src:
            both += 1
        elif src == SOURCE_PMIDB:
            only_m += 1
        else:
            only_p += 1

    stats = {
        "unique_cids": len(order),
        "source_pmidb_only": only_m,
        "source_piazza_only": only_p,
        "source_both": both,
    }
    return records, stats


def merge_protein_sequences(pmidb_path: Path, piazza_path: Path) -> tuple[list[dict], dict]:
    by_id: dict[str, dict] = {}

    def load(path: Path, source: str):
        if not path.is_file():
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = (row.get("Protein_ID") or row.get("protein") or "").strip()
                seq = (row.get("Sequence") or row.get("sequence") or "").strip()
                if not pid:
                    continue
                if pid not in by_id:
                    by_id[pid] = {"Protein_ID": pid, "Sequence": seq, "source": source}
                    continue
                ex = by_id[pid]
                parts = sorted(set(ex["source"].split(";") + source.split(";")))
                ex["source"] = ";".join(parts)
                if len(seq) > len(ex["Sequence"]):
                    ex["Sequence"] = seq

    load(pmidb_path, SOURCE_PMIDB)
    load(piazza_path, SOURCE_PIAZZA)
    rows = sorted(by_id.values(), key=lambda x: x["Protein_ID"])
    return rows, {"unique_proteins": len(rows)}


def write_protein_csv(path: Path, rows: list[dict], with_source: bool = True) -> None:
    fields = ["Protein_ID", "Sequence", "source"] if with_source else ["Protein_ID", "Sequence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def filter_edges_to_nodes(rows: list[dict], node_set: set[str]) -> list[dict]:
    return [r for r in rows if r["node1"] in node_set and r["node2"] in node_set]


def _load_meta_cid_to_row(smi_path: Path) -> dict[str, int]:
    cid_to_row: dict[str, int] = {}
    if not smi_path.is_file():
        return cid_to_row
    with open(smi_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parsed = _parse_meta_smi_line(line)
            if parsed is not None:
                cid_to_row[parsed[1]] = i
    return cid_to_row


def _load_protein_id_to_row(csv_path: Path) -> dict[str, int]:
    id_to_row: dict[str, int] = {}
    if not csv_path.is_file():
        return id_to_row
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pid = (row.get("Protein_ID") or row.get("protein") or "").strip()
            if not pid and row:
                pid = str(next(iter(row.values()))).strip()
            if pid:
                id_to_row[pid] = i
    return id_to_row


def merge_embedding_npy_optional(
    out_dir: Path,
    smi_filtered: list[dict],
    pro_filtered: list[dict],
) -> dict:
    """若 piazza / pmidb 目录存在对应 npy，则按 ID 拼接；否则跳过并记录说明。"""
    report: dict = {"meta": {}, "protein": {}, "skipped": False}

    meta_p = PIAZZA_DIR / "meta_ChemGPT-19M.npy"
    pro_p = PIAZZA_DIR / "protein_large_model.npy"
    meta_m = PMIDB_DIR / "meta_ChemGPT-19M.npy"
    pro_m = PMIDB_DIR / "protein_large_model.npy"

    if not meta_p.is_file() and not meta_m.is_file():
        report["skipped"] = True
        report["reason"] = "no meta_ChemGPT-19M.npy in piazza or PMIDB/ecoil"
        return report

    ref_meta = meta_p if meta_p.is_file() else meta_m
    ref_pro = pro_p if pro_p.is_file() else pro_m
    if not ref_pro.is_file():
        report["skipped"] = True
        report["reason"] = "no protein_large_model.npy in piazza or PMIDB/ecoil"
        return report

    pmidb_meta_map = _load_meta_cid_to_row(PMIDB_DIR / "meta.smi")
    piazza_meta_map = _load_meta_cid_to_row(PIAZZA_DIR / "meta.smi")
    pmidb_pro_map = _load_protein_id_to_row(PMIDB_DIR / "matched_protein_sequences.csv")
    piazza_pro_map = _load_protein_id_to_row(PIAZZA_DIR / "matched_protein_sequences.csv")

    pmidb_meta_npy = np.load(meta_m) if meta_m.is_file() else None
    piazza_meta_npy = np.load(meta_p) if meta_p.is_file() else None
    pmidb_pro_npy = np.load(pro_m) if pro_m.is_file() else None
    piazza_pro_npy = np.load(pro_p) if pro_p.is_file() else None

    ref_meta_npy = pmidb_meta_npy if pmidb_meta_npy is not None else piazza_meta_npy
    ref_pro_npy = pmidb_pro_npy if pmidb_pro_npy is not None else piazza_pro_npy
    dim_m = ref_meta_npy.shape[1]
    dim_p = ref_pro_npy.shape[1]

    meta_rows = []
    meta_from_pmidb = meta_from_piazza = meta_missing = 0
    for rec in smi_filtered:
        cid = rec["cid"]
        if pmidb_meta_npy is not None and cid in pmidb_meta_map:
            meta_rows.append(pmidb_meta_npy[pmidb_meta_map[cid]])
            meta_from_pmidb += 1
        elif piazza_meta_npy is not None and cid in piazza_meta_map:
            meta_rows.append(piazza_meta_npy[piazza_meta_map[cid]])
            meta_from_piazza += 1
        else:
            rng = np.random.default_rng(abs(hash(cid)) % (2**32))
            meta_rows.append(rng.standard_normal(dim_m).astype(np.float32))
            meta_missing += 1

    pro_rows = []
    pro_from_pmidb = pro_from_piazza = pro_missing = 0
    for rec in pro_filtered:
        pid = rec["Protein_ID"]
        if pmidb_pro_npy is not None and pid in pmidb_pro_map:
            pro_rows.append(pmidb_pro_npy[pmidb_pro_map[pid]])
            pro_from_pmidb += 1
        elif piazza_pro_npy is not None and pid in piazza_pro_map:
            pro_rows.append(piazza_pro_npy[piazza_pro_map[pid]])
            pro_from_piazza += 1
        else:
            pro_rows.append(np.random.randn(dim_p).astype(np.float32))
            pro_missing += 1

    meta_arr = np.stack(meta_rows, axis=0)
    pro_arr = np.stack(pro_rows, axis=0)
    np.save(out_dir / "meta_ChemGPT-19M.npy", meta_arr)
    np.save(out_dir / "protein_large_model.npy", pro_arr)

    report["meta"] = {
        "shape": list(meta_arr.shape),
        "from_pmidb": meta_from_pmidb,
        "from_piazza": meta_from_piazza,
        "random_fallback": meta_missing,
    }
    report["protein"] = {
        "shape": list(pro_arr.shape),
        "from_pmidb": pro_from_pmidb,
        "from_piazza": pro_from_piazza,
        "random_fallback": pro_missing,
    }
    return report


def build(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"output_dir": str(out_dir)}

    m_p, m_p_stats = merge_m_p(PMIDB_M_P, PIAZZA_M_P)
    report["m_p_links"] = m_p_stats

    metas = {r["meta"] for r in m_p}
    proteins = {r["protein"] for r in m_p}

    write_m_p(out_dir / "m_p_links_merged.csv", m_p)
    write_m_p(out_dir / "m_p_links.csv", m_p)

    mm_m = read_tsv_edges(PMIDB_DIR / "m_m_links.tsv", SOURCE_PMIDB)
    mm_p = read_tsv_edges(PIAZZA_DIR / "m_m_links.tsv", SOURCE_PIAZZA)
    mm_merged, mm_stats = merge_undirected_edges(mm_m, mm_p)
    mm_filtered = filter_edges_to_nodes(mm_merged, metas)
    report["m_m_links"] = {
        **mm_stats,
        "pmidb_tsv_rows": len(mm_m),
        "piazza_tsv_rows": len(mm_p),
        "after_filter_to_pmi_metas": len(mm_filtered),
        "pmi_meta_count": len(metas),
    }
    write_tsv_with_source(out_dir / "m_m_links_with_source.tsv", mm_merged)
    write_tsv_training(out_dir / "m_m_links.tsv", mm_filtered)

    pp_m = read_tsv_edges(PMIDB_DIR / "p_p_links.tsv", SOURCE_PMIDB)
    pp_p = read_tsv_edges(PIAZZA_DIR / "p_p_links.tsv", SOURCE_PIAZZA)
    pp_merged, pp_stats = merge_undirected_edges(pp_m, pp_p)
    pp_filtered = filter_edges_to_nodes(pp_merged, proteins)
    report["p_p_links"] = {
        **pp_stats,
        "pmidb_tsv_rows": len(pp_m),
        "piazza_tsv_rows": len(pp_p),
        "after_filter_to_pmi_proteins": len(pp_filtered),
        "pmi_protein_count": len(proteins),
    }
    write_tsv_with_source(out_dir / "p_p_links_with_source.tsv", pp_merged)
    write_tsv_training(out_dir / "p_p_links.tsv", pp_filtered)

    smi_records, smi_stats = merge_meta_smi(PMIDB_DIR / "meta.smi", PIAZZA_DIR / "meta.smi")
    smi_filtered = [r for r in smi_records if r["cid"] in metas]
    report["meta_smi"] = {**smi_stats, "lines_in_pmi_metas": len(smi_filtered)}
    with open(out_dir / "meta.smi", "w", encoding="utf-8") as f:
        f.write(
            "\n".join(f"{r['smi']} {r['cid']}" for r in smi_filtered)
            + ("\n" if smi_filtered else "")
        )
    with open(out_dir / "meta_with_source.smi", "w", encoding="utf-8") as f:
        f.write(
            "\n".join(f"{r['smi']} {r['cid']}\t{r['source']}" for r in smi_filtered)
            + ("\n" if smi_filtered else "")
        )

    pro_rows, pro_stats = merge_protein_sequences(
        PMIDB_DIR / "matched_protein_sequences.csv",
        PIAZZA_DIR / "matched_protein_sequences.csv",
    )
    pro_filtered = [r for r in pro_rows if r["Protein_ID"] in proteins]
    report["matched_protein_sequences"] = {**pro_stats, "in_pmi_proteins": len(pro_filtered)}
    write_protein_csv(out_dir / "matched_protein_sequences_with_source.csv", pro_filtered)
    write_protein_csv(out_dir / "matched_protein_sequences.csv", pro_filtered, with_source=False)

    report["embeddings"] = merge_embedding_npy_optional(out_dir, smi_filtered, pro_filtered)

    readme = out_dir / "README_merge.txt"
    readme.write_text(
        "合并数据集 piazza (m_p_links.csv) + PMIDB/ecoil (m_p_links.csv)\n"
        "- m_p_links_merged.csv / m_p_links.csv: 列 meta, protein, target, source\n"
        "- PMIDB 源 CSV 列映射: KEGG->meta, Uniprot_KB_id->protein, interaction->target\n"
        "- m_m_links.tsv / p_p_links.tsv: 三列 node1\\tnode2\\tscore；"
        "来自两侧目录已有 TSV 融合后按 PMI 节点过滤\n"
        "- source 取值: pmidb_ecoil, piazza, 或二者用分号连接\n",
        encoding="utf-8",
    )

    with open(out_dir / "merge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    print("=" * 60)
    print("合并 piazza + PMIDB/ecoil -> piazza_pmidb_ecoil")
    print("=" * 60)
    if not PMIDB_M_P.is_file():
        raise FileNotFoundError(PMIDB_M_P)
    if not PIAZZA_M_P.is_file():
        raise FileNotFoundError(PIAZZA_M_P)

    rep = build(OUT_DIR)
    mp = rep["m_p_links"]
    print(f"\n-> {OUT_DIR.name}")
    print(
        f"  PMI: pmidb={mp['pmidb_rows']}, piazza={mp['piazza_rows']}, "
        f"merged={mp['merged_rows']} (overlap={mp['both_sources']})"
    )
    print(f"  m_m filtered: {rep['m_m_links']['after_filter_to_pmi_metas']}")
    print(f"  p_p filtered: {rep['p_p_links']['after_filter_to_pmi_proteins']}")
    if rep["embeddings"].get("skipped"):
        print(f"  embeddings: skipped ({rep['embeddings'].get('reason', '')})")
    print("\n完成。")


if __name__ == "__main__":
    main()