from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_public_paper_style import PreparedData, SweepConfig, calc_metrics, prepare_data, run_one


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INPUT = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_RESULTS = ROOT / "reproduce" / "public_paper_target_optimization_results.csv"
OUT_BEST = ROOT / "reproduce" / "paper_style_nav_public_paper_target_best.csv"

TARGET_ANN = 0.037
TARGET_SHARPE = 1.27
TARGET_CALMAR = 1.12
TARGET_NONZERO = 0.55


def target_score(metrics: dict[str, float], nonzero_ratio: float) -> float:
    if any(np.isnan(metrics[k]) for k in ["ann", "sharpe", "calmar", "mdd"]):
        return -np.inf
    ann_gap = abs(metrics["ann"] - TARGET_ANN)
    sharpe_gap = abs(metrics["sharpe"] - TARGET_SHARPE)
    calmar_gap = abs(metrics["calmar"] - TARGET_CALMAR) if pd.notna(metrics["calmar"]) else 10.0
    nonzero_gap = abs(nonzero_ratio - TARGET_NONZERO)
    mdd_penalty = max(abs(metrics["mdd"]) - 0.05, 0.0)
    positive_bonus = 0.4 if metrics["ann"] > 0 and metrics["sharpe"] > 0 else -1.0
    return -(2.0 * ann_gap + 1.5 * sharpe_gap + 1.0 * calmar_gap + 0.5 * nonzero_gap + 2.0 * mdd_penalty) + positive_bonus


def build_grid() -> list[SweepConfig]:
    grid = []
    for mom_short, mom_long, threshold, signal_mode, reverse_signal, top_1, top_2, top_3, top_4, cost in product(
        [5, 10, 20],
        [126, 252],
        [0, 1],
        ["or", "and"],
        [False, True],
        [5, 10, 15],
        [5, 10, 15],
        [3, 5, 10],
        [3, 5],
        [0.0, 0.00003, 0.00005],
    ):
        if mom_short >= mom_long:
            continue
        grid.append(
            SweepConfig(
                input_xlsx=PUBLIC_INPUT,
                mom_short=mom_short,
                mom_long=mom_long,
                open_threshold=threshold,
                signal_mode=signal_mode,
                reverse_signal=reverse_signal,
                top_1=top_1,
                top_2=top_2,
                top_3=top_3,
                top_4=top_4,
                cost_per_turn=cost,
            )
        )
    return grid


def main() -> None:
    if not PUBLIC_INPUT.exists():
        raise FileNotFoundError(f"Missing {PUBLIC_INPUT}. Run build_public_macro_dataset.py first.")

    data: PreparedData = prepare_data(SweepConfig(input_xlsx=PUBLIC_INPUT))
    rows: list[dict[str, float | int | str | bool]] = []
    best_score = -np.inf
    best_df = None
    best_cfg = None

    grid = build_grid()
    print(f"total paper-target parameter sets: {len(grid)}")
    for i, cfg in enumerate(grid, start=1):
        df = run_one(cfg, data)
        metrics = calc_metrics(df["StrategyRet"])
        nonzero_ratio = float((df["FinalPos"] != 0).mean())
        score = target_score(metrics, nonzero_ratio)
        rows.append(
            {
                "mom_short": cfg.mom_short,
                "mom_long": cfg.mom_long,
                "threshold": cfg.open_threshold,
                "signal_mode": cfg.signal_mode,
                "reverse_signal": cfg.reverse_signal,
                "top_1": cfg.top_1,
                "top_2": cfg.top_2,
                "top_3": cfg.top_3,
                "top_4": cfg.top_4,
                "cost_per_turn": cfg.cost_per_turn,
                "ann": metrics["ann"],
                "vol": metrics["vol"],
                "sharpe": metrics["sharpe"],
                "mdd": metrics["mdd"],
                "calmar": metrics["calmar"],
                "nonzero_ratio": nonzero_ratio,
                "target_score": score,
                "ann_gap": abs(metrics["ann"] - TARGET_ANN),
                "sharpe_gap": abs(metrics["sharpe"] - TARGET_SHARPE),
                "calmar_gap": abs(metrics["calmar"] - TARGET_CALMAR) if pd.notna(metrics["calmar"]) else np.nan,
            }
        )
        if score > best_score:
            best_score = score
            best_df = df.copy()
            best_cfg = cfg
        if i % 50 == 0:
            print(f"processed {i} paper-target sets; current best score={best_score:.4f}")

    result_df = pd.DataFrame(rows).sort_values(["target_score", "ann_gap", "sharpe_gap"], ascending=[False, True, True])
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_RESULTS, index=False, encoding="utf-8-sig")
    if best_df is None or best_cfg is None:
        raise RuntimeError("No paper-target optimization result generated")
    best_df.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    print("saved", OUT_RESULTS)
    print("saved", OUT_BEST)
    print(best_cfg)
    print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
