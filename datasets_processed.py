"""
processed 数据集配置：链接 CSV 与各数据集在仓库内的目录（meta.smi、.npy 等默认同目录）。
"""
from pathlib import Path

REPO = Path(r"E:/JZT_XIAOLUNWEN")

DATASETS = {
    "stitch_yeast_400": {
        "repo": REPO / "stitch_yeast",
        "links_file": "m_p_links_400_processed.csv",
        "model_dir": "model_stitch_yeast_400_processed",
    },
    "stitch_yeast_700": {
        "repo": REPO / "stitch_yeast",
        "links_file": "m_p_links_700_processed.csv",
        "model_dir": "model_stitch_yeast_700_processed",
    },
    "stitch_ecoli_400": {
        "repo": REPO / "stitch_ecoli",
        "links_file": "m_p_links_400_processed.csv",
        "model_dir": "model_stitch_ecoli_400_processed",
    },
    "stitch_ecoli_700": {
        "repo": REPO / "stitch_ecoli",
        "links_file": "m_p_links_700_processed.csv",
        "model_dir": "model_stitch_ecoli_700_processed",
    },
    "pmidb_human": {
        "repo": REPO / "PMIDB" / "human",
        "links_file": "m_p_links_processed.csv",
        "model_dir": "model_pmidb_human_processed",
    },
    "piazza": {
        "repo": REPO / "piazza",
        "links_file": "m_p_links_processed.csv",
        "model_dir": "model_piazza_processed",
    },
    "meta_links_data": {
        "repo": REPO / "meta_links_data",
        "links_file": "m_p_links.xlsx",
        "model_dir": "model_meta_links_data_processed",
        # 第 3 列 Combined 分数 0–999；> threshold 为正样本
        "label_rule": "combined_score_gt",
        "label_threshold": 400,
    },
}

# 原始 PMI CSV（非 process_m_p_links_sample_neg 生成）
RAW_LINKS_FILE = {
    "stitch_yeast_400": "m_p_links_400.csv",
    "stitch_yeast_700": "m_p_links_700.csv",
    "stitch_ecoli_400": "m_p_links_400.csv",
    "stitch_ecoli_700": "m_p_links_700.csv",
    "pmidb_human": "m_p_links.csv",
    "piazza": "m_p_links.csv",
    "meta_links_data": "m_p_links.xlsx",
}


def get_dataset(name: str, *, processed: bool = True) -> dict:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Choose from: {list(DATASETS)}")
    cfg = dict(DATASETS[name])
    cfg["name"] = name
    cfg["repo"] = Path(cfg["repo"])
    cfg["features_dir"] = cfg["repo"]
    cfg["processed"] = processed
    if not processed:
        cfg["links_file"] = RAW_LINKS_FILE[name]
        base = name if not name.endswith("_processed") else name.replace("_processed", "")
        cfg["model_dir"] = f"model_{base}"
    return cfg