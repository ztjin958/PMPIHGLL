"""生成各数据集 × (protein / meta) 本地 CV 入口脚本。"""
from pathlib import Path

REPO = Path(r"E:/JZT_XIAOLUNWEN")

DATASETS_R10 = [
    "stitch_ecoli_400",
    "stitch_ecoli_700",
    "stitch_yeast_400",
    "stitch_yeast_700",
    "piazza",
]

DATASETS_R1 = [
    "pmidb_human",
]

TEMPLATE_R10 = '''"""
本地 5-fold × 10 RUS：{dataset_key}，按 {split_mode} 节点划分。
用法: conda activate pyjzt && python {script_name}.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from main_local_cv import run_local_cv
from prepare_core import run_prepare

DATASET_KEY = "{dataset_key}"
SPLIT_MODE = "{split_mode}"

if __name__ == "__main__":
    prep = run_prepare(DATASET_KEY, processed=False)
    run_local_cv(DATASET_KEY, SPLIT_MODE, prep=prep, processed=False, R=10)
'''

TEMPLATE_R1 = '''"""
本地 5-fold × 1 RUS（仅 PMIDB）：{dataset_key}，按 {split_mode} 节点划分。
用法: conda activate pyjzt && python {script_name}.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from main_local_cv import run_local_cv
from prepare_core import run_prepare

DATASET_KEY = "{dataset_key}"
SPLIT_MODE = "{split_mode}"

if __name__ == "__main__":
    prep = run_prepare(DATASET_KEY, processed=False)
    run_local_cv(DATASET_KEY, SPLIT_MODE, prep=prep, processed=False, R=1)
'''


def main():
    for key in DATASETS_R10:
        for split in ("protein", "meta"):
            name = f"main_local_{key}_{split}"
            path = REPO / f"{name}.py"
            path.write_text(
                TEMPLATE_R10.format(
                    dataset_key=key,
                    split_mode=split,
                    script_name=name,
                ),
                encoding="utf-8",
            )
            print("wrote", path.name, "(R=10)")
    for key in DATASETS_R1:
        for split in ("protein", "meta"):
            name = f"main_local_{key}_{split}"
            path = REPO / f"{name}.py"
            path.write_text(
                TEMPLATE_R1.format(
                    dataset_key=key,
                    split_mode=split,
                    script_name=name,
                ),
                encoding="utf-8",
            )
            print("wrote", path.name, "(R=1)")


if __name__ == "__main__":
    main()