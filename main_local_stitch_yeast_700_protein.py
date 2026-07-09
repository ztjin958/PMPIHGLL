"""
本地 5-fold × 10 RUS：stitch_yeast_700，按 protein 节点划分。
用法: conda activate pyjzt && python main_local_stitch_yeast_700_protein.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from main_local_cv import run_local_cv
from prepare_core import run_prepare

DATASET_KEY = "stitch_yeast_700"
SPLIT_MODE = "protein"

if __name__ == "__main__":
    prep = run_prepare(DATASET_KEY, processed=False)
    run_local_cv(DATASET_KEY, SPLIT_MODE, prep=prep, processed=False, R=10)
