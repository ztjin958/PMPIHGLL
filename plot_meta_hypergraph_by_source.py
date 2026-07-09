"""
绘制训练用 meta / protein 超图，构图与 prepare_transductive_stitch_piazza.py 一致。

构图依据（与训练时 hyperedge 相同）:
  - Meta: 节点 = 训练+测试 PMI 并集 + meta.smi + m_m_links；相似度矩阵 X 来自 m_m_links.tsv
    经 edgelist_to_matrix 得到的行向量；超边 H = construct_H_with_KNN(X, K=[4,9], metric=cosine)。
  - Protein: 节点 = PMI 并集 + p_p_links + matched_protein_sequences；相似度来自 p_p_links.tsv
    经 edgelist_to_matrix；超边 H = construct_H_with_KNN(X, K=[7,10], metric=cosine)。

节点着色:
  - Meta: features_dir/meta_with_source.smi（stitch_ecoli / piazza / both）
  - Protein: 由 stitch / piazza 的 m_p_links 中是否出现该蛋白推断（无 both 文件时）

输出 (analysis_meta_hypergraph/):
  meta_hypergraph_k{4,9}_{label}.png, meta_hypergraph_bipartite_{label}.png
  protein_hypergraph_k{7,10}_{label}.png, protein_hypergraph_bipartite_{label}.png

用法:
  conda activate pyjzt
  python plot_meta_hypergraph_by_source.py --label 400
  python plot_meta_hypergraph_by_source.py --label 400 --kind protein
  python plot_meta_hypergraph_by_source.py --label 400 --kind both

  # merge_piazza_pmidb_ecoil.py 输出目录 piazza_pmidb_ecoil/
  python plot_meta_hypergraph_by_source.py --preset piazza_pmidb_ecoil --kind both
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

from utils import construct_H_with_KNN, edgelist_to_matrix

REPO = Path(r"E:/JZT_XIAOLUNWEN")
SOURCE_STITCH = "stitch_ecoli"
SOURCE_PIAZZA = "piazza"
SOURCE_PMIDB = "pmidb_ecoil"
MERGE_DIR = REPO / "piazza_pmidb_ecoil"

COLORS = {
    SOURCE_STITCH: "#2E86AB",
    SOURCE_PIAZZA: "#E94F37",
    SOURCE_PMIDB: "#1B998B",
    "both": "#7B2D8E",
    "unknown": "#888888",
}


def _label_to_int(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in ("1", "True", "true", "TRUE"):
        return 1
    if s in ("0", "False", "false", "FALSE"):
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def read_pmi_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=[df.columns[0], df.columns[1]], how="any")
    c0, c1 = df.columns[0], df.columns[1]
    label_col = next(
        (n for n in ("target", "score", "label") if n in df.columns),
        df.columns[2],
    )
    out = pd.DataFrame(
        {
            "meta": df[c0].astype(str).str.strip(),
            "protein": df[c1].astype(str).str.strip(),
            "label": df[label_col].map(_label_to_int),
        }
    )
    return out.drop_duplicates(subset=["meta", "protein"], keep="first")


def normalize_source_tag(src: str) -> str:
    """merge 输出 pmidb_ecoil;piazza → both；stitch_ecoli 等保持。"""
    s = (src or "").strip()
    if not s:
        return "unknown"
    if ";" in s:
        parts = {p.strip() for p in s.split(";") if p.strip()}
        if parts == {SOURCE_PMIDB, SOURCE_PIAZZA}:
            return "both"
        if len(parts) > 1:
            return "both"
    return s


def load_meta_source(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                body, src = line.rsplit("\t", 1)
                src = normalize_source_tag(src.strip())
            else:
                body = line
                src = "unknown"
            parts = body.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            cid = parts[1].strip()
            out[cid] = src or "unknown"
    return out


def read_merged_pmi_csv(path: Path) -> pd.DataFrame:
    """merge 后的 m_p_links.csv（含 source 列）。"""
    df = pd.read_csv(path)
    df = df.dropna(subset=["meta", "protein"], how="any")
    for col in ("meta", "protein"):
        df[col] = df[col].astype(str).str.strip()
    label_col = next(
        (n for n in ("target", "score", "label") if n in df.columns),
        df.columns[2] if len(df.columns) > 2 else "target",
    )
    src_col = "source" if "source" in df.columns else None
    out = pd.DataFrame(
        {
            "meta": df["meta"],
            "protein": df["protein"],
            "label": df[label_col].map(_label_to_int),
        }
    )
    if src_col:
        out["source"] = df[src_col].astype(str).map(normalize_source_tag)
    else:
        out["source"] = "unknown"
    return out.drop_duplicates(subset=["meta", "protein"], keep="first")


def meta_source_from_merged_pmi(pmi_path: Path, meta_list: list[str]) -> dict[str, str]:
    df = read_merged_pmi_csv(pmi_path)
    by_meta: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        m, s = row["meta"], row["source"]
        by_meta.setdefault(m, set()).add(s)
    out: dict[str, str] = {}
    for m in meta_list:
        tags = by_meta.get(m, set())
        tags.discard("unknown")
        if not tags:
            out[m] = "unknown"
        elif len(tags) == 1:
            out[m] = next(iter(tags))
        else:
            out[m] = "both"
    return out


def protein_source_from_merged_pmi(pmi_path: Path, protein_list: list[str]) -> dict[str, str]:
    df = read_merged_pmi_csv(pmi_path)
    by_pro: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        p, s = row["protein"], row["source"]
        by_pro.setdefault(p, set()).add(s)
    out: dict[str, str] = {}
    for p in protein_list:
        tags = by_pro.get(p, set())
        tags.discard("unknown")
        if not tags:
            out[p] = "unknown"
        elif len(tags) == 1:
            out[p] = next(iter(tags))
        else:
            out[p] = "both"
    return out


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


def _collect_proteins_from_seq(path: Path) -> set[str]:
    s: set[str] = set()
    if not path.is_file():
        return s
    df = pd.read_csv(path)
    col = "Protein_ID" if "Protein_ID" in df.columns else df.columns[0]
    for v in df[col].astype(str):
        s.add(v.strip())
    return s


def protein_source_from_pmi(
    stitch_links: Path, piazza_links: Path, protein_list: list[str]
) -> dict[str, str]:
    """训练/测试 PMI 中出现的蛋白 → stitch_ecoli / piazza / both。"""
    df_st = read_pmi_csv(stitch_links)
    df_pz = read_pmi_csv(piazza_links)
    in_st = set(df_st["protein"].astype(str))
    in_pz = set(df_pz["protein"].astype(str))
    out: dict[str, str] = {}
    for p in protein_list:
        a, b = p in in_st, p in in_pz
        if a and b:
            out[p] = "both"
        elif a:
            out[p] = SOURCE_STITCH
        elif b:
            out[p] = SOURCE_PIAZZA
        else:
            out[p] = "unknown"
    return out


def build_meta_list_from_merged_pmi(
    merged_pmi: Path, features_dir: Path
) -> list[str]:
    df = read_merged_pmi_csv(merged_pmi)
    metas = set(df["meta"])
    smi = features_dir / "meta.smi"
    if smi.is_file():
        with open(smi, encoding="utf-8") as f:
            for line in f:
                p = line.strip().rsplit(" ", 1)
                if len(p) == 2:
                    metas.add(p[1].strip())
    mm = features_dir / "m_m_links.tsv"
    if mm.is_file():
        with open(mm, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    metas.add(parts[0].strip())
                    metas.add(parts[1].strip())
    return sorted(metas)


def build_protein_list_from_merged_pmi(
    merged_pmi: Path, features_dir: Path
) -> list[str]:
    df = read_merged_pmi_csv(merged_pmi)
    proteins = set(df["protein"])
    pp_a, pp_b = _collect_ids_from_tsv(features_dir / "p_p_links.tsv")
    proteins |= pp_a | pp_b
    proteins |= _collect_proteins_from_seq(
        features_dir / "matched_protein_sequences.csv"
    )
    return sorted(proteins)


def build_meta_list(stitch_links: Path, piazza_links: Path, features_dir: Path) -> list[str]:
    df_st = read_pmi_csv(stitch_links)
    df_pz = read_pmi_csv(piazza_links)
    metas = set(df_st["meta"]) | set(df_pz["meta"])
    smi = features_dir / "meta.smi"
    if smi.is_file():
        with open(smi, encoding="utf-8") as f:
            for line in f:
                p = line.strip().rsplit(" ", 1)
                if len(p) == 2:
                    metas.add(p[1].strip())
    mm = features_dir / "m_m_links.tsv"
    if mm.is_file():
        with open(mm, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    metas.add(parts[0].strip())
                    metas.add(parts[1].strip())
    return sorted(metas)


def build_meta_sim_matrix(meta_list: list[str], features_dir: Path) -> np.ndarray:
    """与 prepare 中 X_meta_sim 一致。"""
    path = str(features_dir)
    meta_dict = {m: i for i, m in enumerate(meta_list)}
    n = len(meta_list)

    df_meta_edgelist = pd.read_csv(
        os.path.join(path, "m_m_links.tsv"),
        header=None,
        sep="\t",
        names=["node1", "node2", "weight"],
    )
    meta_edgelist = os.path.join(path, "meta_edge_plot.edgelist")
    with open(meta_edgelist, "w", encoding="utf-8") as f:
        for i in range(len(df_meta_edgelist)):
            n1 = str(df_meta_edgelist.iloc[i, 0]).strip()
            n2 = str(df_meta_edgelist.iloc[i, 1]).strip()
            if n1 not in meta_dict or n2 not in meta_dict:
                continue
            w = df_meta_edgelist.iloc[i, 2]
            f.write(f"{meta_dict[n1]} {meta_dict[n2]} {w}\n")

    emb = edgelist_to_matrix(n, meta_edgelist)
    if emb is None:
        emb = np.eye(n, dtype=np.float64)
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i < emb.shape[0]:
            X[i] = emb[i]
        else:
            X[i, i] = 1.0
    return X


def build_protein_list(
    stitch_links: Path, piazza_links: Path, features_dir: Path
) -> list[str]:
    """与 prepare_transductive 中 df_pmi_protein_list 一致。"""
    df_st = read_pmi_csv(stitch_links)
    df_pz = read_pmi_csv(piazza_links)
    proteins = set(df_st["protein"]) | set(df_pz["protein"])
    pp_a, pp_b = _collect_ids_from_tsv(features_dir / "p_p_links.tsv")
    proteins |= pp_a | pp_b
    proteins |= _collect_proteins_from_seq(
        features_dir / "matched_protein_sequences.csv"
    )
    return sorted(proteins)


def build_protein_sim_matrix(
    protein_list: list[str], features_dir: Path
) -> np.ndarray:
    """与 prepare 中 X_protein_sim / sim_protein_features 一致（p_p edgelist 嵌入）。"""
    path = str(features_dir)
    protein_dict = {p: i for i, p in enumerate(protein_list)}
    n = len(protein_list)

    df_protein_edgelist = pd.read_csv(
        os.path.join(path, "p_p_links.tsv"),
        header=None,
        sep="\t",
        names=["node1", "node2", "weight"],
    )
    protein_edgelist = os.path.join(path, "protein_edge_plot.edgelist")
    with open(protein_edgelist, "w", encoding="utf-8") as f:
        for i in range(len(df_protein_edgelist)):
            n1 = str(df_protein_edgelist.iloc[i, 0]).strip()
            n2 = str(df_protein_edgelist.iloc[i, 1]).strip()
            if n1 not in protein_dict or n2 not in protein_dict:
                continue
            w = df_protein_edgelist.iloc[i, 2]
            f.write(f"{protein_dict[n1]} {protein_dict[n2]} {w}\n")

    emb = edgelist_to_matrix(n, protein_edgelist)
    if emb is None:
        emb = np.eye(n, dtype=np.float64)
    X = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i < emb.shape[0]:
            X[i] = emb[i]
        else:
            X[i, i] = 1.0
    return X


def hypergraph_to_meta_clique_graph(H: np.ndarray, k_label: int) -> nx.Graph:
    """每条超边（列）内节点两两连边，边属性记录超边 id。"""
    G = nx.Graph()
    n_nodes, n_edges = H.shape
    for v in range(n_nodes):
        G.add_node(v)
    for e in range(n_edges):
        members = np.where(H[:, e] > 0)[0]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u, w = int(members[i]), int(members[j])
                if G.has_edge(u, w):
                    G[u][w].setdefault("hyperedges", set()).add(e)
                else:
                    G.add_edge(u, w, hyperedges={e}, k_scale=k_label)
    return G


def short_cid(cid: str, n: int = 10) -> str:
    s = cid.replace("CIDs", "")
    if len(s) > n:
        return s[-n:]
    return s


def short_protein_id(pid: str, n: int = 10) -> str:
    s = str(pid).strip()
    if len(s) > n:
        return s[-n:]
    return s


def layout_meta_graph(G: nx.Graph, seed: int = 42) -> dict:
    if G.number_of_edges() == 0:
        return nx.circular_layout(G)
    try:
        pos = nx.spring_layout(G, seed=seed, k=1.8, iterations=200)
    except Exception:
        pos = nx.kamada_kawai_layout(G)
    return pos


def _source_tag(src: str) -> str:
    if src == SOURCE_STITCH:
        return "S"
    if src == SOURCE_PIAZZA:
        return "P"
    if src == SOURCE_PMIDB:
        return "M"
    if src == "both":
        return "B"
    return "?"


def _legend_patches(preset: str) -> list:
    if preset == "piazza_pmidb_ecoil":
        return [
            mpatches.Patch(color=COLORS[SOURCE_PMIDB], label="PMIDB/ecoil"),
            mpatches.Patch(color=COLORS[SOURCE_PIAZZA], label="Piazza"),
            mpatches.Patch(color=COLORS["both"], label="both (pmidb_ecoil;piazza)"),
            mpatches.Patch(color=COLORS["unknown"], label="graph only / unknown"),
        ]
    return [
        mpatches.Patch(color=COLORS[SOURCE_STITCH], label="STITCH (stitch_ecoli)"),
        mpatches.Patch(color=COLORS[SOURCE_PIAZZA], label="Piazza"),
        mpatches.Patch(color=COLORS["both"], label="both sources"),
        mpatches.Patch(color=COLORS["unknown"], label="PMI graph only / unknown"),
    ]


def draw_hypergraph_clique(
    node_list: list[str],
    node_source: dict[str, str],
    H: np.ndarray,
    k_neig: int,
    out_path: Path,
    title: str,
    *,
    label_fn=short_cid,
    draw_node_labels: bool = True,
    legend_preset: str = "stitch",
):
    G = hypergraph_to_meta_clique_graph(H, k_neig)
    pos = layout_meta_graph(G)
    n_nodes = len(node_list)
    node_size = 520 if n_nodes <= 80 else (280 if n_nodes <= 200 else 120)
    font_size = 7 if n_nodes <= 80 else (5 if n_nodes <= 200 else 0)

    node_colors = [
        COLORS.get(node_source.get(nid, "unknown"), COLORS["unknown"])
        for nid in node_list
    ]

    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
    nx.draw_networkx_edges(
        G, pos, alpha=0.35, width=1.2, edge_color="#666666", ax=ax
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="white",
        linewidths=1.2,
        ax=ax,
    )

    if draw_node_labels and font_size > 0:
        labels = {i: label_fn(node_list[i]) for i in range(len(node_list))}
        nx.draw_networkx_labels(
            G, pos, labels, font_size=font_size, font_weight="bold", ax=ax
        )

    if n_nodes <= 120:
        for i, nid in enumerate(node_list):
            x, y = pos[i]
            ax.annotate(
                _source_tag(node_source.get(nid, "?")),
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=6,
                color="#333333",
            )

    ax.legend(
        handles=_legend_patches(legend_preset),
        loc="upper left",
        framealpha=0.9,
        fontsize=8,
    )
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_bipartite_hypergraph(
    node_list: list[str],
    node_source: dict[str, str],
    H_list: list[np.ndarray],
    k_labels: list[int],
    out_path: Path,
    *,
    node_prefix: str = "m",
    label_fn=short_cid,
    title: str | None = None,
    legend_left: str = "STITCH",
    legend_right: str = "Piazza",
    legend_preset: str = "stitch",
):
    """左侧实体节点，右侧超边节点（各 k 一列）。"""
    B = nx.Graph()
    entity_nodes = [f"{node_prefix}{i}" for i in range(len(node_list))]
    B.add_nodes_from(entity_nodes, bipartite=0)

    hyper_nodes = []
    for H, k in zip(H_list, k_labels):
        n_nodes, n_edges = H.shape
        for e in range(n_edges):
            members = np.where(H[:, e] > 0)[0]
            if len(members) < 2:
                continue
            he = f"h{k}_{e}"
            hyper_nodes.append(he)
            B.add_node(he, bipartite=1, k_scale=k)
            for v in members:
                B.add_edge(f"{node_prefix}{int(v)}", he)

    pos = {}
    n_m = len(node_list)
    for i, mn in enumerate(entity_nodes):
        y = (i - n_m / 2) / max(n_m, 1)
        pos[mn] = (-1.0, y)

    for k in k_labels:
        he_k = [h for h in hyper_nodes if h.startswith(f"h{k}_")]
        x = 0.35 if k == k_labels[0] else 1.0
        for j, he in enumerate(he_k):
            y = (j - len(he_k) / 2) / max(len(he_k), 1)
            pos[he] = (x, y)

    fig, ax = plt.subplots(figsize=(16, max(10, n_m * 0.22)), dpi=140)
    entity_colors = [
        COLORS.get(node_source.get(nid, "unknown"), COLORS["unknown"])
        for nid in node_list
    ]

    nx.draw_networkx_edges(B, pos, alpha=0.25, width=0.8, ax=ax)
    entity_ns = [n for n in B.nodes() if str(n).startswith(node_prefix)]
    hyper_ns = [n for n in B.nodes() if str(n).startswith("h")]
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=entity_ns,
        node_color=entity_colors,
        node_size=400 if n_m <= 150 else 180,
        edgecolors="white",
        linewidths=1,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=hyper_ns,
        node_color="#F4A261",
        node_size=80,
        alpha=0.85,
        ax=ax,
    )
    if n_m <= 120:
        labels = {
            f"{node_prefix}{i}": label_fn(node_list[i], 8)
            for i in range(len(node_list))
        }
        nx.draw_networkx_labels(B, pos, labels, font_size=7, ax=ax)

    k_str = " & ".join(str(k) for k in k_labels)
    patches = list(_legend_patches(legend_preset))
    patches.append(mpatches.Patch(color="#F4A261", label="hyperedge"))
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.set_title(
        title
        or f"Hypergraph (bipartite): nodes <-> hyperedges (KNN k={k_str})",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _print_source_counts(
    name: str,
    node_list: list[str],
    source: dict[str, str],
    *,
    preset: str = "stitch",
) -> None:
    print(f"[plot] {name}={len(node_list)}")
    keys = (
        (SOURCE_PMIDB, SOURCE_PIAZZA, "both", "unknown")
        if preset == "piazza_pmidb_ecoil"
        else (SOURCE_STITCH, SOURCE_PIAZZA, "both", "unknown")
    )
    for src in keys:
        c = sum(1 for n in node_list if source.get(n) == src)
        if c:
            print(f"  {src}: {c}")


def run_meta(
    label: str,
    features_dir: Path,
    stitch_links: Path,
    piazza_links: Path,
    out_dir: Path,
) -> None:
    meta_list = build_meta_list(stitch_links, piazza_links, features_dir)
    meta_source = load_meta_source(features_dir / "meta_with_source.smi")
    for m in meta_list:
        meta_source.setdefault(m, "unknown")

    _print_source_counts("metas", meta_list, meta_source)

    sim_meta = build_meta_sim_matrix(meta_list, features_dir)
    K_meta = [4, 9]
    H_list = construct_H_with_KNN(sim_meta, K_meta, metric="cosine")

    for k, H in zip(K_meta, H_list):
        out = out_dir / f"meta_hypergraph_k{k}_{label}.png"
        draw_hypergraph_clique(
            meta_list,
            meta_source,
            H,
            k,
            out,
            title=(
                f"Meta hypergraph (KNN k={k}, sim from m_m_links, cosine) "
                "— color by meta_with_source.smi"
            ),
            label_fn=short_cid,
        )
        print(f"[plot] saved {out}")

    bip = out_dir / f"meta_hypergraph_bipartite_{label}.png"
    draw_bipartite_hypergraph(
        meta_list,
        meta_source,
        H_list,
        K_meta,
        bip,
        node_prefix="m",
        label_fn=short_cid,
        title="Meta hypergraph (bipartite): meta nodes <-> hyperedges (KNN k=4 & k=9)",
        legend_left="STITCH meta",
        legend_right="Piazza meta",
    )
    print(f"[plot] saved {bip}")

    rows = [
        {
            "meta_idx": i,
            "meta": cid,
            "source": meta_source.get(cid, "unknown"),
            "label_short": short_cid(cid),
        }
        for i, cid in enumerate(meta_list)
    ]
    pd.DataFrame(rows).to_csv(out_dir / f"meta_nodes_source_{label}.csv", index=False)


def run_protein(
    label: str,
    features_dir: Path,
    stitch_links: Path,
    piazza_links: Path,
    out_dir: Path,
) -> None:
    protein_list = build_protein_list(stitch_links, piazza_links, features_dir)
    protein_source = protein_source_from_pmi(stitch_links, piazza_links, protein_list)

    _print_source_counts("proteins", protein_list, protein_source)

    sim_protein = build_protein_sim_matrix(protein_list, features_dir)
    K_protein = [7, 10]
    H_list = construct_H_with_KNN(sim_protein, K_protein, metric="cosine")

    for k, H in zip(K_protein, H_list):
        out = out_dir / f"protein_hypergraph_k{k}_{label}.png"
        draw_hypergraph_clique(
            protein_list,
            protein_source,
            H,
            k,
            out,
            title=(
                f"Protein hypergraph (KNN k={k}, sim from p_p_links, cosine) "
                "— color by train/test PMI appearance"
            ),
            label_fn=short_protein_id,
            draw_node_labels=len(protein_list) <= 200,
        )
        print(f"[plot] saved {out}")

    bip = out_dir / f"protein_hypergraph_bipartite_{label}.png"
    draw_bipartite_hypergraph(
        protein_list,
        protein_source,
        H_list,
        K_protein,
        bip,
        node_prefix="p",
        label_fn=short_protein_id,
        title="Protein hypergraph (bipartite): protein nodes <-> hyperedges (KNN k=7 & k=10)",
        legend_left="STITCH train PMI",
        legend_right="Piazza test PMI",
    )
    print(f"[plot] saved {bip}")

    rows = [
        {
            "protein_idx": i,
            "protein": pid,
            "source": protein_source.get(pid, "unknown"),
            "label_short": short_protein_id(pid),
        }
        for i, pid in enumerate(protein_list)
    ]
    pd.DataFrame(rows).to_csv(out_dir / f"protein_nodes_source_{label}.csv", index=False)


def run_piazza_pmidb_ecoil(
    features_dir: Path,
    merged_pmi: Path,
    out_dir: Path,
    kind: str,
    tag: str,
) -> None:
    """merge_piazza_pmidb_ecoil.py 输出：m_m/p_p 为融合后 TSV，节点着色来自 PMI source 列。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not merged_pmi.is_file():
        raise FileNotFoundError(merged_pmi)
    if not (features_dir / "m_m_links.tsv").is_file():
        raise FileNotFoundError(features_dir / "m_m_links.tsv")

    preset = "piazza_pmidb_ecoil"
    print(f"[plot] preset=piazza_pmidb_ecoil features={features_dir}")

    if kind in ("meta", "both"):
        meta_list = build_meta_list_from_merged_pmi(merged_pmi, features_dir)
        meta_source = load_meta_source(features_dir / "meta_with_source.smi")
        for m in meta_list:
            if meta_source.get(m, "unknown") == "unknown":
                meta_source[m] = meta_source_from_merged_pmi(merged_pmi, [m]).get(
                    m, "unknown"
                )
        _print_source_counts("metas", meta_list, meta_source, preset=preset)

        sim_meta = build_meta_sim_matrix(meta_list, features_dir)
        K_meta = [4, 9]
        H_list = construct_H_with_KNN(sim_meta, K_meta, metric="cosine")
        for k, H in zip(K_meta, H_list):
            out = out_dir / f"meta_hypergraph_k{k}_{tag}.png"
            draw_hypergraph_clique(
                meta_list,
                meta_source,
                H,
                k,
                out,
                title=(
                    f"[piazza_pmidb_ecoil] Meta hypergraph (KNN k={k}, merged m_m_links, cosine)"
                ),
                label_fn=short_cid,
                legend_preset=preset,
            )
            print(f"[plot] saved {out}")
        bip = out_dir / f"meta_hypergraph_bipartite_{tag}.png"
        draw_bipartite_hypergraph(
            meta_list,
            meta_source,
            H_list,
            K_meta,
            bip,
            node_prefix="m",
            label_fn=short_cid,
            title="[piazza_pmidb_ecoil] Meta bipartite (KNN k=4 & k=9)",
            legend_preset=preset,
        )
        print(f"[plot] saved {bip}")
        pd.DataFrame(
            [
                {
                    "meta_idx": i,
                    "meta": cid,
                    "source": meta_source.get(cid, "unknown"),
                    "label_short": short_cid(cid),
                }
                for i, cid in enumerate(meta_list)
            ]
        ).to_csv(out_dir / f"meta_nodes_source_{tag}.csv", index=False)

    if kind in ("protein", "both"):
        protein_list = build_protein_list_from_merged_pmi(merged_pmi, features_dir)
        protein_source = protein_source_from_merged_pmi(merged_pmi, protein_list)
        _print_source_counts("proteins", protein_list, protein_source, preset=preset)

        sim_protein = build_protein_sim_matrix(protein_list, features_dir)
        K_protein = [7, 10]
        H_list = construct_H_with_KNN(sim_protein, K_protein, metric="cosine")
        for k, H in zip(K_protein, H_list):
            out = out_dir / f"protein_hypergraph_k{k}_{tag}.png"
            draw_hypergraph_clique(
                protein_list,
                protein_source,
                H,
                k,
                out,
                title=(
                    f"[piazza_pmidb_ecoil] Protein hypergraph (KNN k={k}, merged p_p_links, cosine)"
                ),
                label_fn=short_protein_id,
                draw_node_labels=len(protein_list) <= 200,
                legend_preset=preset,
            )
            print(f"[plot] saved {out}")
        bip = out_dir / f"protein_hypergraph_bipartite_{tag}.png"
        draw_bipartite_hypergraph(
            protein_list,
            protein_source,
            H_list,
            K_protein,
            bip,
            node_prefix="p",
            label_fn=short_protein_id,
            title="[piazza_pmidb_ecoil] Protein bipartite (KNN k=7 & k=10)",
            legend_preset=preset,
        )
        print(f"[plot] saved {bip}")
        pd.DataFrame(
            [
                {
                    "protein_idx": i,
                    "protein": pid,
                    "source": protein_source.get(pid, "unknown"),
                    "label_short": short_protein_id(pid),
                }
                for i, pid in enumerate(protein_list)
            ]
        ).to_csv(out_dir / f"protein_nodes_source_{tag}.csv", index=False)


def run(
    label: str,
    features_dir: Path,
    stitch_links: Path,
    piazza_links: Path,
    out_dir: Path,
    kind: str = "both",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind in ("meta", "both"):
        run_meta(label, features_dir, stitch_links, piazza_links, out_dir)
    if kind in ("protein", "both"):
        run_protein(label, features_dir, stitch_links, piazza_links, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Plot meta/protein KNN hypergraphs (same H as prepare_transductive_stitch_piazza)."
    )
    parser.add_argument(
        "--preset",
        choices=("stitch", "piazza_pmidb_ecoil"),
        default="stitch",
        help="piazza_pmidb_ecoil: merge_piazza_pmidb_ecoil.py 输出目录",
    )
    parser.add_argument("--label", choices=("400", "700"), default="400")
    parser.add_argument(
        "--kind",
        choices=("meta", "protein", "both"),
        default="meta",
        help="meta (default, k=4,9); protein (k=7,10); both",
    )
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--stitch-links", type=Path, default=None)
    parser.add_argument("--piazza-links", type=Path, default=None)
    parser.add_argument(
        "--merged-pmi",
        type=Path,
        default=None,
        help="合并 PMI（preset=piazza_pmidb_ecoil 时默认 piazza_pmidb_ecoil/m_p_links.csv）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.preset == "piazza_pmidb_ecoil":
        features = args.features_dir or MERGE_DIR
        merged = args.merged_pmi or (features / "m_p_links.csv")
        out = args.out_dir or (REPO / "analysis_meta_hypergraph" / "piazza_pmidb_ecoil")
        run_piazza_pmidb_ecoil(
            features, merged, out, args.kind, tag="piazza_pmidb_ecoil"
        )
        return

    label = args.label
    features = args.features_dir or (REPO / f"stitch_piazza_ecoli_{label}")
    stitch = args.stitch_links or (REPO / "stitch_ecoli" / f"m_p_links_{label}.csv")
    piazza = args.piazza_links or (REPO / "piazza" / "m_p_links.csv")
    out = args.out_dir or (REPO / "analysis_meta_hypergraph")
    run(label, features, stitch, piazza, out, kind=args.kind)


if __name__ == "__main__":
    main()