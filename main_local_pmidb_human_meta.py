"""
本地 5-fold × 1 RUS（仅 PMIDB）：pmidb_human，按 meta 节点划分。
用法: conda activate pyjzt && python main_local_pmidb_human_meta.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from main_local_cv import run_local_cv
from prepare_core import run_prepare

DATASET_KEY = "pmidb_human"
SPLIT_MODE = "meta"

if __name__ == "__main__":
    prep = run_prepare(DATASET_KEY, processed=False)
    run_local_cv(DATASET_KEY, SPLIT_MODE, prep=prep, processed=False, R=1)
