"""
piazza-pmidb + --rus-test：扫描 drop-test-only-proteins ∈ {0.1,…,0.9}，
每个比例重复训练 N 次，汇总 ACC / F1 / AUC / AUPR 的 mean±std（保留 3 位小数）。

用法（pyjzt）:
  python sweep_drop_test_only_proteins.py
  python sweep_drop_test_only_proteins.py --repeats 10 --out results_drop_protein_sweep.xlsx
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent

# 在 import 训练模块前设置
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from main_stitch_train_piazza_test import run_stitch_train_piazza_test
from prepare_transductive_stitch_piazza import default_prepare_piazza_train_pmidb_test

METRIC_KEYS = (
    ("accuracy", "ACC"),
    ("f1", "F1"),
    ("roc_auc", "AUC"),
    ("aupr", "AUPR"),
)


def _fmt_mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return f"{arr.mean():.3f}±{0.0:.3f}"
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    return f"{m:.3f}±{s:.3f}"


def _run_one(
    prep: dict,
    *,
    run_seed: int,
    quiet: bool,
) -> dict[str, float]:
    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        return run_stitch_train_piazza_test(
            prep,
            use_rus=True,
            rus_test=True,
            train_tag="Piazza",
            test_tag="PMIDB/ecoil",
            run_seed=run_seed,
            quiet=quiet,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="每个 drop 比例重复训练次数（默认 10）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "sweep_drop_test_only_proteins_results.xlsx",
        help="结果 XLSX 路径（含 summary / raw 两个 sheet）",
    )
    parser.add_argument(
        "--frac-start",
        type=float,
        default=0,
        help="起始比例（含）",
    )
    parser.add_argument(
        "--frac-end",
        type=float,
        default=0,
        help="结束比例（含）",
    )
    parser.add_argument(
        "--frac-step",
        type=float,
        default=0.1,
        help="比例步长",
    )
    parser.add_argument(
        "--verbose-prepare",
        action="store_true",
        help="打印 prepare 日志",
    )
    args = parser.parse_args()

    fracs = []
    f = args.frac_start
    while f <= args.frac_end + 1e-9:
        fracs.append(round(f, 10))
        f += args.frac_step

    rows_out: list[dict[str, str]] = []
    raw_rows: list[dict] = []

    print(
        f"[sweep] mode=piazza-pmidb, rus_test=True, repeats={args.repeats}, "
        f"drop_test_only_proteins={fracs}",
        flush=True,
    )

    for frac in fracs:
        print(f"\n[sweep] drop_test_only_proteins={frac:.1f} — prepare …", flush=True)
        prep_ctx = (
            contextlib.nullcontext()
            if args.verbose_prepare
            else contextlib.redirect_stdout(io.StringIO())
        )
        with prep_ctx:
            prep = default_prepare_piazza_train_pmidb_test(
                drop_test_only_protein_fraction=frac,
                drop_test_only_meta_fraction=0.0,
            )

        bucket: dict[str, list[float]] = {k: [] for k, _ in METRIC_KEYS}
        for rep in range(args.repeats):
            run_seed = 1000 * int(round(frac * 10)) + rep
            print(
                f"  run {rep + 1}/{args.repeats} (seed={run_seed}) …",
                flush=True,
            )
            metrics = _run_one(prep, run_seed=run_seed, quiet=True)
            raw_rows.append({"drop_frac": frac, "rep": rep, "seed": run_seed, **metrics})
            for key, _ in METRIC_KEYS:
                bucket[key].append(float(metrics[key]))

        row = {"drop_frac": f"{frac:.1f}"}
        for key, col in METRIC_KEYS:
            row[col] = _fmt_mean_std(bucket[key])
        rows_out.append(row)

    # 打印表格
    cols = ["drop_frac"] + [c for _, c in METRIC_KEYS]
    header = "\t".join(cols)
    print("\n" + "=" * 72)
    print(header)
    for row in rows_out:
        print("\t".join(row[c] for c in cols))
    print("=" * 72)

    # 写 XLSX（summary + raw 两个 sheet）
    out_path = args.out
    if out_path.suffix.lower() != ".xlsx":
        out_path = out_path.with_suffix(".xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_cols = ["drop_frac", "rep", "seed"] + [k for k, _ in METRIC_KEYS]
    try:
        import pandas as pd

        df_sum = pd.DataFrame(rows_out)[cols]
        df_raw = pd.DataFrame(raw_rows)[raw_cols]
        for k, _ in METRIC_KEYS:
            df_raw[k] = df_raw[k].astype(float).round(6)
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_sum.to_excel(writer, sheet_name="summary", index=False)
            df_raw.to_excel(writer, sheet_name="raw", index=False)
    except ImportError as e:
        raise SystemExit(
            "写入 XLSX 需要 pandas 与 openpyxl：\n"
            "  conda activate pyjzt\n"
            "  pip install openpyxl\n"
            f"（原始错误: {e}）"
        ) from e

    print(f"\n[sweep] saved XLSX: {out_path}")
    print("  sheet 'summary': mean±std (ACC, F1, AUC, AUPR)")
    print("  sheet 'raw': 每轮原始指标")


if __name__ == "__main__":
    main()