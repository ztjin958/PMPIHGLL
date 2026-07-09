import csv
from pathlib import Path

base = Path(r"E:/JZT_XIAOLUNWEN")


def ids_from_m_p(path):
    metas, proteins = set(), set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metas.add(row["meta"].strip())
            proteins.add(row["protein"].strip())
    return metas, proteins


def ids_from_tsv(path, col0, col1):
    a, b = set(), set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                a.add(parts[col0].strip())
                b.add(parts[col1].strip())
    return a | b


def ids_from_smi(path):
    s = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) == 2:
                s.add(parts[1].strip())
    return s


p_m, p_p = ids_from_m_p(base / "piazza" / "m_p_links.csv")
s_m_pmi, s_p_pmi = ids_from_m_p(base / "stitch_ecoli" / "m_p_links_400.csv")
s_m_mm = ids_from_tsv(base / "stitch_ecoli" / "m_m_links.tsv", 0, 1)
s_m_all = s_m_pmi | s_m_mm | ids_from_smi(base / "stitch_ecoli" / "meta.smi")
s_p_pp = ids_from_tsv(base / "stitch_ecoli" / "p_p_links.tsv", 0, 1)
s_p_all = s_p_pmi | s_p_pp

print("Piazza meta", len(p_m), "protein", len(p_p))
print("STITCH meta PMI only", len(s_m_pmi), "meta all files", len(s_m_all))
print("STITCH protein PMI only", len(s_p_pmi), "protein all files", len(s_p_all))
print("meta overlap PMI", len(p_m & s_m_pmi), "meta overlap all", len(p_m & s_m_all))
print("protein overlap PMI", len(p_p & s_p_pmi), "protein overlap all", len(p_p & s_p_all))

try:
    from rdkit import Chem

    def canon_set(smi_path):
        out = set()
        with open(smi_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                m = Chem.MolFromSmiles(parts[0])
                if m:
                    out.add(Chem.MolToSmiles(m, canonical=True))
        return out

    pc, sc = canon_set(base / "piazza" / "meta.smi"), canon_set(base / "stitch_ecoli" / "meta.smi")
    print("canonical SMILES overlap", len(pc & sc))
except Exception as e:
    print("RDKit:", e)