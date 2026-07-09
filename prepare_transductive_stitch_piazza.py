"""
STITCH 训练 / Piazza 测试（传导式）：
- 节点：STITCH 与 Piazza 全部代谢物、蛋白（并集）
- 超图：在全部节点上构图（与 main.py / prepare_core 相同 KNN 超边）
- 监督边：训练仅用 stitch_ecoli 的 PMI；测试仅用 piazza 的 PMI（默认去掉与 STITCH 重复的 (meta,protein)）
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from utils import *

np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

REPO = Path(r"E:/JZT_XIAOLUNWEN")


def _label_to_int(val):
    if pd.isna(val):
        return 0
    if isinstance(val, (bool, np.bool_)):
        return int(val)
    s = str(val).strip()
    if s in ("1", "True", "true", "TRUE"):
        return 1
    if s in ("0", "False", "false", "FALSE"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _read_pmi_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[df.columns[0], df.columns[1]], how="any")
    c0, c1 = df.columns[0], df.columns[1]
    label_col = None
    for name in ("target", "score", "label"):
        if name in df.columns:
            label_col = name
            break
    if label_col is None:
        label_col = df.columns[2]
    out = pd.DataFrame(
        {
            "meta": df[c0].astype(str).str.strip(),
            "protein": df[c1].astype(str).str.strip(),
            "label": df[label_col].map(_label_to_int),
        }
    )
    return out.drop_duplicates(subset=["meta", "protein"], keep="first")


def _collect_ids_from_tsv(path: Path) -> tuple[set[str], set[str]]:
    a, b = set(), set()
    if not path.is_file():
        return a, b
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                a.add(parts[0].strip())
                b.add(parts[1].strip())
    return a, b


def _parse_meta_smi_line(line: str) -> tuple[str, str] | None:
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


def _collect_meta_from_smi(path: Path) -> set[str]:
    s = set()
    if not path.is_file():
        return s
    with open(path, encoding="utf-8") as f:
        for line in f:
            parsed = _parse_meta_smi_line(line)
            if parsed is not None:
                s.add(parsed[1])
    return s


def _collect_proteins_from_seq(path: Path) -> set[str]:
    s = set()
    if not path.is_file():
        return s
    df = pd.read_csv(path)
    col = "Protein_ID" if "Protein_ID" in df.columns else df.columns[0]
    for v in df[col].astype(str):
        s.add(v.strip())
    return s


def _select_test_only_ids_to_drop(
    test_rows: list[tuple[str, str, int]],
    train_ids: set[str],
    drop_fraction: float,
    *,
    which: str,
    rng_seed: int,
) -> set[str]:
    """
    在「当前测试边」里，找出训练 PMI 未出现过的 meta 或 protein，随机剔除 drop_fraction 比例。
    which: "meta" -> 取边第一列；"protein" -> 第二列。从测试边与构图节点中一并删除。
    """
    if drop_fraction <= 0 or not test_rows:
        return set()
    if which == "meta":
        ids_in_test = {m for (m, _, _) in test_rows}
    elif which == "protein":
        ids_in_test = {p for (_, p, _) in test_rows}
    else:
        raise ValueError(which)
    test_only = sorted(ids_in_test - train_ids)
    if not test_only:
        return set()
    n_drop = int(len(test_only) * drop_fraction)
    if n_drop <= 0:
        return set()
    rng = np.random.default_rng(rng_seed)
    chosen = rng.choice(test_only, size=n_drop, replace=False)
    return set(chosen.tolist())


def run_prepare_transductive(
    *,
    stitch_links: Path,
    piazza_links: Path,
    features_dir: Path,
    exclude_test_overlap: bool = True,
    drop_test_only_protein_fraction: float = 0.0,
    drop_test_only_meta_fraction: float = 0.0,
    label: str = "400",
    train_name: str = "train",
    test_name: str = "test",
) -> dict:
    """
    stitch_links / piazza_links：分别为训练、测试 PMI 文件路径（历史命名，与 STITCH/Piazza 无必然对应）。
    features_dir: 合并后的目录（如 stitch_piazza_ecoli_400），含 meta.smi、p_p、m_m、npy 等。
    """
    if not stitch_links.is_file():
        raise FileNotFoundError(stitch_links)
    if not piazza_links.is_file():
        raise FileNotFoundError(piazza_links)
    path = str(features_dir)

    df_stitch = _read_pmi_csv(stitch_links)
    df_piazza = _read_pmi_csv(piazza_links)

    stitch_keys = set(zip(df_stitch["meta"], df_stitch["protein"]))

    metas: set[str] = set(df_stitch["meta"]) | set(df_piazza["meta"])
    proteins: set[str] = set(df_stitch["protein"]) | set(df_piazza["protein"])

    metas |= _collect_meta_from_smi(features_dir / "meta.smi")
    mm_a, mm_b = _collect_ids_from_tsv(features_dir / "m_m_links.tsv")
    metas |= mm_a | mm_b
    pp_a, pp_b = _collect_ids_from_tsv(features_dir / "p_p_links.tsv")
    proteins |= pp_a | pp_b
    proteins |= _collect_proteins_from_seq(
        features_dir / "matched_protein_sequences.csv"
    )

    train_meta_ids = set(df_stitch["meta"].astype(str).str.strip())
    train_protein_ids = set(df_stitch["protein"].astype(str).str.strip())

    test_rows: list[tuple[str, str, int]] = []
    overlap_test = 0
    for _, row in df_piazza.iterrows():
        m, p = row["meta"], row["protein"]
        if exclude_test_overlap and (m, p) in stitch_keys:
            overlap_test += 1
            continue
        if m not in metas or p not in proteins:
            continue
        test_rows.append((m, p, int(row["label"])))

    n_test_before_node_filter = len(test_rows)
    metas_test_only_before = len({m for (m, _, _) in test_rows} - train_meta_ids)
    proteins_test_only_before = len({p for (_, p, _) in test_rows} - train_protein_ids)

    meta_ids_to_drop = _select_test_only_ids_to_drop(
        test_rows,
        train_meta_ids,
        drop_test_only_meta_fraction,
        which="meta",
        rng_seed=43,
    )
    n_meta_graph_removed = len(meta_ids_to_drop)
    if meta_ids_to_drop:
        metas -= meta_ids_to_drop
        test_rows = [(m, p, y) for m, p, y in test_rows if m not in meta_ids_to_drop]

    protein_ids_to_drop = _select_test_only_ids_to_drop(
        test_rows,
        train_protein_ids,
        drop_test_only_protein_fraction,
        which="protein",
        rng_seed=42,
    )
    n_prot_graph_removed = len(protein_ids_to_drop)
    if protein_ids_to_drop:
        proteins -= protein_ids_to_drop
        test_rows = [(m, p, y) for m, p, y in test_rows if p not in protein_ids_to_drop]

    n_edges_node_filter_dropped = n_test_before_node_filter - len(test_rows)

    df_pmi_meta_list = sorted(metas)
    df_pmi_protein_list = sorted(proteins)
    meta_dict = {m: i for i, m in enumerate(df_pmi_meta_list)}
    protein_dict = {p: i for i, p in enumerate(df_pmi_protein_list)}

    print(
        f"[transductive] train ({train_name}) PMI={len(df_stitch)}, "
        f"test ({test_name}) PMI={len(df_piazza)}"
    )
    print(f"[transductive] train_links={stitch_links}")
    print(f"[transductive] test_links={piazza_links}")
    print(
        f"[transductive] union nodes: meta={len(df_pmi_meta_list)}, "
        f"protein={len(df_pmi_protein_list)}"
    )
    print(f"[transductive] features_dir={features_dir}")

    df_protein_edgelist = pd.read_csv(
        os.path.join(path, "p_p_links.tsv"),
        header=None,
        sep="\t",
        names=["node1", "node2", "weight"],
    )
    df_meta_edgelist = pd.read_csv(
        os.path.join(path, "m_m_links.tsv"),
        header=None,
        sep="\t",
        names=["node1", "node2", "weight"],
    )

    protein_edgelist = os.path.join(path, "protein_edge.edgelist")
    with open(protein_edgelist, "w") as f:
        for i in range(len(df_protein_edgelist)):
            try:
                n1 = df_protein_edgelist.iloc[i, 0]
                n2 = df_protein_edgelist.iloc[i, 1]
                if n1 not in protein_dict or n2 not in protein_dict:
                    continue
                f.write(
                    f"{protein_dict[n1]} {protein_dict[n2]} "
                    f"{df_protein_edgelist.iloc[i, 2]}\n"
                )
            except Exception:
                pass

    meta_edgelist = os.path.join(path, "meta_edge.edgelist")
    with open(meta_edgelist, "w") as f:
        for i in range(len(df_meta_edgelist)):
            try:
                n1 = df_meta_edgelist.iloc[i, 0]
                n2 = df_meta_edgelist.iloc[i, 1]
                if n1 not in meta_dict or n2 not in meta_dict:
                    continue
                f.write(
                    f"{meta_dict[n1]} {meta_dict[n2]} "
                    f"{df_meta_edgelist.iloc[i, 2]}\n"
                )
            except Exception:
                pass

    df_embedding_protein = edgelist_to_matrix(
        len(df_pmi_protein_list), protein_edgelist
    )
    df_embedding_meta = edgelist_to_matrix(len(df_pmi_meta_list), meta_edgelist)
    protein_name_df = pd.read_csv(os.path.join(path, "matched_protein_sequences.csv"))
    df_meta = pd.read_csv(os.path.join(path, "meta.smi"), sep=" ", header=None)

    X_protein_large_model = {}
    df_protein_large_model = np.load(os.path.join(path, "protein_large_model.npy"))
    pro_count = 0
    for key, value in protein_dict.items():
        try:
            index_in_folder = protein_name_df.loc[
                protein_name_df.iloc[:, 0] == key
            ].index
            X_protein_large_model[value] = df_protein_large_model[
                index_in_folder.item()
            ]
        except Exception:
            X_protein_large_model[value] = np.random.rand(
                df_protein_large_model.shape[1]
            ).flatten()
            pro_count += 1
    if pro_count:
        print(f"Not in protein_large_model:{pro_count}")

    X_protein_sim = {}
    pro_count = 0
    for key, value in protein_dict.items():
        try:
            X_protein_sim[value] = df_embedding_protein[value]
        except Exception:
            X_protein_sim[value] = np.random.rand(len(df_pmi_protein_list)).flatten()
            pro_count += 1
    if pro_count:
        print(f"Not in X_protein_sim:{pro_count}")

    df_meta_large_model = np.load(os.path.join(path, "meta_ChemGPT-19M.npy"))
    X_meta_large_model = {}
    pro_count = 0
    for key, value in meta_dict.items():
        try:
            index = df_meta[df_meta.iloc[:, 1] == key].index[0]
            X_meta_large_model[value] = df_meta_large_model[index]
        except Exception:
            X_meta_large_model[value] = np.random.rand(
                df_meta_large_model.shape[1]
            ).flatten()
            pro_count += 1
    if pro_count:
        print(f"Not in meta_large_model:{pro_count}")

    X_meta_sim = {}
    pro_count = 0
    for key, value in meta_dict.items():
        try:
            X_meta_sim[value] = df_embedding_meta[value]
        except Exception:
            X_meta_sim[value] = np.random.rand(len(df_pmi_meta_list)).flatten()
            pro_count += 1
    if pro_count:
        print(f"Not in embeddings_meta:{pro_count}")

    K_meta = [4, 9]
    K_protein = [7, 10]

    protein_large_model_features = np.array(
        [X_protein_large_model[i] for i in range(len(df_pmi_protein_list))]
    )
    meta_large_model_features = np.array(
        [X_meta_large_model[i] for i in range(len(df_pmi_meta_list))]
    )

    sim_protein_features = np.array(
        [X_protein_sim[i] for i in range(len(df_pmi_protein_list))]
    )
    H_protein_dis_knn_from_sim = construct_H_with_KNN(
        sim_protein_features, K_protein, metric="cosine"
    )
    hyperedge_protein_dis_knn_from_sim_index = convert_adjacency_matrix(
        H_protein_dis_knn_from_sim
    )

    sim_meta_features = np.array(
        [X_meta_sim[i] for i in range(len(df_pmi_meta_list))]
    )
    H_meta_dis_knn_from_sim = construct_H_with_KNN(
        sim_meta_features, K_meta, metric="cosine"
    )
    hyperedge_meta_dis_knn_from_sim_index = convert_adjacency_matrix(
        H_meta_dis_knn_from_sim
    )

    train_meta_idx = []
    train_protein_idx = []
    train_y = []
    skipped_train = 0
    for _, row in df_stitch.iterrows():
        m, p = row["meta"], row["protein"]
        if m not in meta_dict or p not in protein_dict:
            skipped_train += 1
            continue
        train_meta_idx.append(meta_dict[m])
        train_protein_idx.append(protein_dict[p])
        train_y.append(int(row["label"]))

    skipped_test = 0
    for _, row in df_piazza.iterrows():
        m, p = row["meta"], row["protein"]
        if exclude_test_overlap and (m, p) in stitch_keys:
            continue
        if m not in meta_dict or p not in protein_dict:
            skipped_test += 1

    test_meta_idx = []
    test_protein_idx = []
    test_y = []
    for m, p, lab in test_rows:
        test_meta_idx.append(meta_dict[m])
        test_protein_idx.append(protein_dict[p])
        test_y.append(lab)

    n_pos_train = sum(train_y)
    n_pos_test = sum(test_y)
    print(
        f"[transductive] train edges={len(train_y)} (pos={n_pos_train}), "
        f"skipped={skipped_train}  [main 默认 RUS 训练集为 1:1，见 train samples=]"
    )
    print(
        f"[transductive] test edges={len(test_y)} (pos={n_pos_test}), "
        f"skipped={skipped_test}, overlap_removed={overlap_test}"
    )
    if drop_test_only_meta_fraction > 0:
        print(
            f"[transductive] test-only metas (before filter)={metas_test_only_before}; "
            f"removed {n_meta_graph_removed} meta nodes from graph "
            f"({drop_test_only_meta_fraction:.0%} of test-only)"
        )
    if drop_test_only_protein_fraction > 0:
        print(
            f"[transductive] test-only proteins (before filter)={proteins_test_only_before}; "
            f"removed {n_prot_graph_removed} protein nodes from graph "
            f"({drop_test_only_protein_fraction:.0%} of test-only)"
        )
    if drop_test_only_meta_fraction > 0 or drop_test_only_protein_fraction > 0:
        print(
            f"[transductive] test edges after node drop: "
            f"{n_test_before_node_filter} -> {len(test_y)} "
            f"(total removed {n_edges_node_filter_dropped})"
        )
    print("Prepare transductive is done")

    cfg = {
        "name": f"stitch_train_piazza_test_{label}",
        "model_dir": f"model_stitch_train_piazza_test_{label}",
        "stitch_links": str(stitch_links),
        "piazza_links": str(piazza_links),
        "features_dir": str(features_dir),
    }

    return {
        "device": device,
        "cfg": cfg,
        "meta_list": df_pmi_meta_list,
        "protein_list": df_pmi_protein_list,
        "sim_meta_features": sim_meta_features,
        "sim_protein_features": sim_protein_features,
        "meta_large_model_features": meta_large_model_features,
        "protein_large_model_features": protein_large_model_features,
        "hyperedge_protein_dis_knn_from_sim_index": hyperedge_protein_dis_knn_from_sim_index,
        "hyperedge_meta_dis_knn_from_sim_index": hyperedge_meta_dis_knn_from_sim_index,
        "train_meta_idx": np.array(train_meta_idx, dtype=np.int64),
        "train_protein_idx": np.array(train_protein_idx, dtype=np.int64),
        "train_Y": np.array(train_y, dtype=np.int64),
        "test_meta_idx": np.array(test_meta_idx, dtype=np.int64),
        "test_protein_idx": np.array(test_protein_idx, dtype=np.int64),
        "test_Y": np.array(test_y, dtype=np.int64),
    }


def default_prepare_400(
    stitch_processed: bool = False,
    piazza_processed: bool = False,
    drop_test_only_protein_fraction: float = 0.0,
    drop_test_only_meta_fraction: float = 0.0,
) -> dict:
    suffix = "_processed" if stitch_processed else ""
    psuffix = "_processed" if piazza_processed else ""
    stitch = REPO / "stitch_ecoli" / f"m_p_links_400{suffix}.csv"
    if not stitch.is_file():
        stitch = REPO / "stitch_ecoli" / "m_p_links_400.csv"
    piazza = REPO / "piazza" / f"m_p_links{psuffix}.csv"
    if not piazza.is_file():
        piazza = REPO / "piazza" / "m_p_links.csv"
    features = REPO / "stitch_piazza_ecoli_400"
    return run_prepare_transductive(
        stitch_links=stitch,
        piazza_links=piazza,
        features_dir=features,
        label="400",
        train_name="STITCH",
        test_name="Piazza",
        drop_test_only_protein_fraction=drop_test_only_protein_fraction,
        drop_test_only_meta_fraction=drop_test_only_meta_fraction,
    )


def default_prepare_700(
    stitch_processed: bool = False,
    piazza_processed: bool = False,
    drop_test_only_protein_fraction: float = 0.0,
    drop_test_only_meta_fraction: float = 0.0,
) -> dict:
    suffix = "_processed" if stitch_processed else ""
    psuffix = "_processed" if piazza_processed else ""
    stitch = REPO / "stitch_ecoli" / f"m_p_links_700{suffix}.csv"
    if not stitch.is_file():
        stitch = REPO / "stitch_ecoli" / "m_p_links_700.csv"
    piazza = REPO / "piazza" / f"m_p_links{psuffix}.csv"
    if not piazza.is_file():
        piazza = REPO / "piazza" / "m_p_links.csv"
    features = REPO / "stitch_piazza_ecoli_700"
    return run_prepare_transductive(
        stitch_links=stitch,
        piazza_links=piazza,
        features_dir=features,
        label="700",
        train_name="STITCH",
        test_name="Piazza",
        drop_test_only_protein_fraction=drop_test_only_protein_fraction,
        drop_test_only_meta_fraction=drop_test_only_meta_fraction,
    )


def default_prepare_piazza_train_pmidb_test(
    *,
    piazza_processed: bool = False,
    pmidb_processed: bool = False,
    drop_test_only_protein_fraction: float = 0.0,
    drop_test_only_meta_fraction: float = 0.0,
) -> dict:
    """
    训练：piazza/m_p_links.csv；测试：PMIDB/ecoil/m_p_links.csv；
    特征与 m_m / p_p：piazza_pmidb_ecoil（merge_piazza_pmidb_ecoli.py 输出）。
    """
    psuffix = "_processed" if piazza_processed else ""
    msuffix = "_processed" if pmidb_processed else ""
    train_links = REPO / "piazza" / f"m_p_links{psuffix}.csv"
    if not train_links.is_file():
        train_links = REPO / "piazza" / "m_p_links.csv"
    test_links = REPO / "PMIDB" / "ecoil" / f"m_p_links{msuffix}.csv"
    if not test_links.is_file():
        test_links = REPO / "PMIDB" / "ecoil" / "m_p_links.csv"
    features = REPO / "piazza_pmidb_ecoil"
    return run_prepare_transductive(
        stitch_links=train_links,
        piazza_links=test_links,
        features_dir=features,
        label="piazza_train_pmidb_test",
        train_name="Piazza",
        test_name="PMIDB/ecoil",
        drop_test_only_protein_fraction=drop_test_only_protein_fraction,
        drop_test_only_meta_fraction=drop_test_only_meta_fraction,
    )