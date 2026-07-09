"""
基于 Prepare.py：读取 *_processed.csv，构建特征与超图，供 main_core 训练。
"""
import os
import shutil

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from datasets_processed import get_dataset
from utils import *

np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_p_p_links_tsv(features_dir: str) -> str:
    """PMIDB 等目录可能只有 p_p_links_part1.tsv …，合并为 p_p_links.tsv。"""
    out = os.path.join(features_dir, "p_p_links.tsv")
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return out
    parts = sorted(
        f
        for f in os.listdir(features_dir)
        if f.startswith("p_p_links_part") and f.endswith(".tsv")
    )
    if not parts:
        return out
    print(f"[prepare] merging {len(parts)} p_p_links_part*.tsv -> p_p_links.tsv")
    with open(out, "wb") as w:
        for name in parts:
            with open(os.path.join(features_dir, name), "rb") as r:
                shutil.copyfileobj(r, w)
    return out


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


def _pmi_edge_label(val, cfg: dict) -> int:
    """第三列 → 0/1。meta_links_data：Combined 分数，> label_threshold 为正。"""
    rule = cfg.get("label_rule")
    if rule == "combined_score_gt":
        thr = float(cfg.get("label_threshold", 400))
        if pd.isna(val):
            return 0
        try:
            return 1 if float(val) > thr else 0
        except (TypeError, ValueError):
            return 0
    return _label_to_int(val)


def _cap_proteins_equal_meta_count(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """
    PMI 中蛋白种类数 > 代谢物种类数时：按 PMI 边出现次数（度）从高到低保留与代谢物同数量的蛋白，
    仅保留两端均在保留蛋白集合内的边。
    """
    c0, c1 = df.columns[0], df.columns[1]
    meta_series = df[c0].astype(str).str.strip()
    prot_series = df[c1].astype(str).str.strip()
    n_meta = meta_series.nunique()
    deg = prot_series.value_counts()
    protein_ids = list(deg.index)
    n_prot = len(protein_ids)
    if n_prot <= n_meta:
        print(
            f"[prepare] cap_protein_to_meta: proteins={n_prot} <= meta={n_meta}, skip"
        )
        return df
    rng = np.random.RandomState(seed)
    tie = {p: rng.random() for p in protein_ids}
    ranked = sorted(protein_ids, key=lambda p: (-int(deg[p]), tie[p]))
    keep = set(ranked[:n_meta])
    out = df.loc[prot_series.isin(keep)].copy()
    print(
        f"[prepare] cap_protein_to_meta: proteins {n_prot} -> {n_meta}, "
        f"PMI rows {len(df)} -> {len(out)}"
    )
    return out


def run_prepare(
    dataset_key: str,
    *,
    processed: bool = True,
    combined_threshold: float | None = None,
    cap_protein_to_meta: bool = False,
    protein_cap_seed: int = 42,
) -> dict:
    cfg = get_dataset(dataset_key, processed=processed)
    if combined_threshold is not None:
        if cfg.get("label_rule") != "combined_score_gt":
            print(
                f"[prepare] warning: --combined-threshold={combined_threshold} "
                f"ignored (dataset {dataset_key!r} has no combined_score_gt rule)"
            )
        else:
            cfg["label_threshold"] = float(combined_threshold)
            print(f"[prepare] Combined label threshold overridden → {cfg['label_threshold']}")
    repo = cfg["repo"]
    path = str(cfg["features_dir"])
    links_path = repo / cfg["links_file"]
    if not links_path.is_file():
        raise FileNotFoundError(links_path)

    print(f"[prepare] dataset={dataset_key}")
    print(f"[prepare] links={links_path}")
    print(f"[prepare] features_dir={path}")

    if str(links_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(links_path)
    else:
        df = pd.read_csv(links_path)
    df = df.dropna(subset=[df.columns[0], df.columns[1]], how="any")
    if cap_protein_to_meta:
        df = _cap_proteins_equal_meta_count(df, seed=protein_cap_seed)
        cfg = dict(cfg)
        cfg["cap_protein_to_meta"] = True
        cfg["protein_cap_seed"] = protein_cap_seed
    edges_df = df.iloc[:, [0, 1]]

    protein_dict = {}
    meta_dict = {}

    df_pmi_meta_list = df.iloc[:, 0].drop_duplicates().values.tolist()
    df_pmi_protein_list = df.iloc[:, 1].drop_duplicates().values.tolist()
    for idx, id in enumerate(df_pmi_protein_list):
        protein_dict[id] = idx
    for idx, id in enumerate(df_pmi_meta_list):
        meta_dict[id] = idx

    edges_out = os.path.join(path, "edges.csv")
    with open(edges_out, "w") as f:
        for i in range(len(edges_df)):
            w = _pmi_edge_label(df.iloc[i, 2], cfg)
            f.write(
                f"{meta_dict[edges_df.iloc[i, 0]]},{protein_dict[edges_df.iloc[i, 1]]},{w}\n"
            )

    n_pos = sum(_pmi_edge_label(df.iloc[i, 2], cfg) for i in range(len(df)))
    if cfg.get("label_rule") == "combined_score_gt":
        thr = cfg.get("label_threshold", 400)
        print(
            f"[prepare] PMI rows={len(df)}, Combined>{thr} → positive={n_pos}, "
            f"negative={len(df) - n_pos}"
        )
    else:
        print(f"[prepare] PMI rows={len(df)}, positive(1/True)={n_pos}")

    df_edges = pd.read_csv(edges_out, header=None, names=["node1", "node2", "weight"])
    df_meta = pd.read_csv(os.path.join(path, "meta.smi"), sep=" ", header=None)

    p_p_path = _ensure_p_p_links_tsv(path)
    df_protein_edgelist = pd.read_csv(
        p_p_path,
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
    if cap_protein_to_meta:
        protein_edgelist = os.path.join(path, "protein_edge_eq_meta.edgelist")
    rebuild_protein_el = cap_protein_to_meta or not os.path.exists(protein_edgelist)
    if rebuild_protein_el:
        with open(protein_edgelist, "w") as f:
            for i in range(len(df_protein_edgelist)):
                try:
                    f.write(
                        f"{protein_dict[df_protein_edgelist.iloc[i, 0]]} "
                        f"{protein_dict[df_protein_edgelist.iloc[i, 1]]} "
                        f"{df_protein_edgelist.iloc[i, 2]}\n"
                    )
                except Exception:
                    pass

    meta_edgelist = os.path.join(path, "meta_edge.edgelist")
    if cap_protein_to_meta:
        meta_edgelist = os.path.join(path, "meta_edge_eq_meta.edgelist")
    rebuild_meta_el = cap_protein_to_meta or not os.path.exists(meta_edgelist)
    if rebuild_meta_el:
        with open(meta_edgelist, "w") as f:
            for i in range(len(df_meta_edgelist)):
                try:
                    f.write(
                        f"{meta_dict[df_meta_edgelist.iloc[i, 0]]} "
                        f"{meta_dict[df_meta_edgelist.iloc[i, 1]]} "
                        f"{df_meta_edgelist.iloc[i, 2]}\n"
                    )
                except Exception:
                    pass

    df_embedding_protein = edgelist_to_matrix(
        len(df_pmi_protein_list), protein_edgelist
    )
    df_embedding_meta = edgelist_to_matrix(len(df_pmi_meta_list), meta_edgelist)
    protein_name_df = pd.read_csv(os.path.join(path, "matched_protein_sequences.csv"))

    X_protein_large_model = {}
    df_protein_large_model = np.load(os.path.join(path, "protein_large_model.npy"))
    pro_count = 0
    for key, value in protein_dict.items():
        try:
            index_in_folder = protein_name_df.loc[protein_name_df.iloc[:, 0] == key].index
            X_protein_large_model[value] = df_protein_large_model[index_in_folder.item()]
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

    X_proteins = df_edges.iloc[:, 1].values.tolist()
    X_metas = df_edges.iloc[:, 0].values.tolist()
    Y = np.array(df_edges.iloc[:, 2].values.tolist())

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

    H5_IS_PROBH = True
    H5_M_PROB = 1.0
    H_protein_dis_knn_weighted = construct_H_with_KNN(
        sim_protein_features,
        K_protein,
        metric="cosine",
        is_probH=H5_IS_PROBH,
        m_prob=H5_M_PROB,
    )
    (
        hyperedge_protein_weighted_index,
        hyperedge_protein_weighted_incidence,
        hyperedge_protein_weighted_he_weight,
    ) = convert_hypergraph_to_pyg(H_protein_dis_knn_weighted)

    sim_meta_features = np.array([X_meta_sim[i] for i in range(len(df_pmi_meta_list))])
    H_meta_dis_knn_from_sim = construct_H_with_KNN(
        sim_meta_features, K_meta, metric="cosine"
    )
    hyperedge_meta_dis_knn_from_sim_index = convert_adjacency_matrix(
        H_meta_dis_knn_from_sim
    )

    H_meta_dis_knn_weighted = construct_H_with_KNN(
        sim_meta_features,
        K_meta,
        metric="cosine",
        is_probH=H5_IS_PROBH,
        m_prob=H5_M_PROB,
    )
    (
        hyperedge_meta_weighted_index,
        hyperedge_meta_weighted_incidence,
        hyperedge_meta_weighted_he_weight,
    ) = convert_hypergraph_to_pyg(H_meta_dis_knn_weighted)

    score_lists = {
        "recall_scores": [],
        "aupr_scores": [],
        "f1_scores": [],
        "mcc_scores": [],
        "prescision_scores": [],
        "roc_scores": [],
        "precision_scores": [],
        "precision_recall_scores": [],
        "roc_auc_scores": [],
        "accuracy_scores": [],
        "specificity_scores": [],
        "precision_score_scores": [],
    }

    print(f"meta_nums:{len(df_pmi_meta_list)}  protein_nums:{len(df_pmi_protein_list)}")
    print("Prepare is done")

    out = {
        "device": device,
        "cfg": cfg,
        "meta_large_model_features": meta_large_model_features,
        "protein_large_model_features": protein_large_model_features,
        "hyperedge_protein_dis_knn_from_sim_index": hyperedge_protein_dis_knn_from_sim_index,
        "hyperedge_meta_dis_knn_from_sim_index": hyperedge_meta_dis_knn_from_sim_index,
        "X_proteins": X_proteins,
        "X_metas": X_metas,
        "Y": Y,
        "meta_id_list": list(df_pmi_meta_list),
        "protein_id_list": list(df_pmi_protein_list),
        **score_lists,
    }
    return out