from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_public_paper_style import PreparedData, SweepConfig, calc_metrics, prepare_data, run_one


ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_RESULTS = ROOT / "reproduce" / "public_fourth_stage_robust_results.csv"
OUT_BEST = ROOT / "reproduce" / "paper_style_nav_public_fourth_stage_robust_best.csv"


@dataclass(frozen=True)
class Fixed:
    mom_short: int
    mom_long: int
    threshold: int
    reverse_signal: bool
    top_1: int
    top_2: int
    top_3: int
    top_4: int


def load_second_stage_best_params() -> dict[str, object]:
    path = ROOT / "reproduce" / "public_second_stage_refine_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run second_stage_refine.py first.")

    best = pd.read_csv(path).sort_values(["target_score"], ascending=False).iloc[0]
    return {
        "mom_short": int(best["mom_short"]),
        "mom_long": int(best["mom_long"]),
        "threshold": int(best["threshold"]),
        "reverse_signal": bool(best["reverse_signal"]),
        "top_1": int(best["top_1"]),
        "top_2": int(best["top_2"]),
        "top_3": int(best["top_3"]),
        "top_4": int(best["top_4"]),
    }


def robust_objective(sharpes: list[float]) -> float:
    # maximize the worst-case sharpe across different costs
    if not sharpes:
        return -np.inf
    return float(np.min(sharpes))


def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Missing {INPUT_XLSX}. Run build_public_macro_dataset.py first.")

    base = load_second_stage_best_params()
    fixed = Fixed(
        mom_short=int(base["mom_short"]),
        mom_long=int(base["mom_long"]),
        threshold=int(base["threshold"]),
        reverse_signal=bool(base["reverse_signal"]),
        top_1=int(base["top_1"]),
        top_2=int(base["top_2"]),
        top_3=int(base["top_3"]),
        top_4=int(base["top_4"]),
    )

    data: PreparedData = prepare_data(SweepConfig(input_xlsx=INPUT_XLSX))

    signal_modes = ["or"]
    z_win_levels = [126, 252]
    z_win_diffs = [63, 126]
    z_minp_levels = [60, 90]
    z_minp_diffs = [20]

    stress_costs = [0.0, 0.00003, 0.00005]

    total = len(signal_modes) * len(z_win_levels) * len(z_win_diffs) * len(z_minp_levels) * len(z_minp_diffs)
    print(f"fourth-stage robust combos: {total} (each evaluated on {len(stress_costs)} costs)")

    rows: list[dict[str, object]] = []
    best_score = float("-inf")
    best_df = None
    best_cfg = None

    for i, (signal_mode, z_win_level, z_win_diff, z_minp_level, z_minp_diff) in enumerate(
        product(signal_modes, z_win_levels, z_win_diffs, z_minp_levels, z_minp_diffs),
        start=1,
    ):
        sharpes: list[float] = []
        anns: list[float] = []
        mdds: list[float] = []

        best_cost_df = None
        best_cost_cfg = None
        best_cost_sharpe = float("-inf")

        for cost in stress_costs:
            cfg = SweepConfig(
                input_xlsx=INPUT_XLSX,
                mom_short=fixed.mom_short,
                mom_long=fixed.mom_long,
                open_threshold=fixed.threshold,
                signal_mode=str(signal_mode),
                reverse_signal=fixed.reverse_signal,
                top_1=fixed.top_1,
                top_2=fixed.top_2,
                top_3=fixed.top_3,
                top_4=fixed.top_4,
                z_win_level=int(z_win_level),
                z_minp_level=int(z_minp_level),
                z_win_diff=int(z_win_diff),
                z_minp_diff=int(z_minp_diff),
                cost_per_turn=float(cost),
            )
            df = run_one(cfg, data)
            metrics = calc_metrics(df["StrategyRet"])
            sharpes.append(float(metrics["sharpe"]))
            anns.append(float(metrics["ann"]))
            mdds.append(float(metrics["mdd"]))

            if metrics["sharpe"] > best_cost_sharpe:
                best_cost_sharpe = float(metrics["sharpe"])
                best_cost_df = df.copy()
                best_cost_cfg = cfg

        score = robust_objective(sharpes)

        rows.append(
            {
                "signal_mode": str(signal_mode),
                "z_win_level": int(z_win_level),
                "z_minp_level": int(z_minp_level),
                "z_win_diff": int(z_win_diff),
                "z_minp_diff": int(z_minp_diff),
                "mom_short": fixed.mom_short,
                "mom_long": fixed.mom_long,
                "threshold": fixed.threshold,
                "reverse_signal": fixed.reverse_signal,
                "top_1": fixed.top_1,
                "top_2": fixed.top_2,
                "top_3": fixed.top_3,
                "top_4": fixed.top_4,
                "robust_min_sharpe": score,
                "sharpe_cost0": sharpes[0],
                "sharpe_cost3": sharpes[1],
                "sharpe_cost5": sharpes[2],
                "ann_cost0": anns[0],
                "ann_cost3": anns[1],
                "ann_cost5": anns[2],
                "mdd_cost0": mdds[0],
                "mdd_cost3": mdds[1],
                "mdd_cost5": mdds[2],
            }
        )

        if score > best_score:
            best_score = score
            best_df = best_cost_df
            best_cfg = best_cost_cfg

        if i % 5 == 0:
            print(f"processed {i}/{total}; current best robust_min_sharpe={best_score:.4f}")

    out = pd.DataFrame(rows).sort_values(["robust_min_sharpe", "sharpe_cost0"], ascending=False)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_RESULTS, index=False, encoding="utf-8-sig")

    if best_df is None or best_cfg is None:
        raise RuntimeError("No fourth-stage robust result generated")
    best_df.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    print("saved", OUT_RESULTS)
    print("saved", OUT_BEST)
    print("best cfg (representative, best-by-sharpe among costs):")
    print(best_cfg)
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
