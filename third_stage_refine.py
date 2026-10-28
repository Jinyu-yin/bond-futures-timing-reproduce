from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_public_paper_style import PreparedData, SweepConfig, calc_metrics, prepare_data, run_one


ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_RESULTS = ROOT / "reproduce" / "public_third_stage_results.csv"
OUT_BEST = ROOT / "reproduce" / "paper_style_nav_public_third_stage_best.csv"

TARGET_ANN = 0.037
TARGET_SHARPE = 1.27
TARGET_CALMAR = 1.12
TARGET_NONZERO = 0.55


def load_second_stage_best_params() -> dict[str, object]:
    path = ROOT / "reproduce" / "public_second_stage_refine_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run second_stage_refine.py first.")

    best = pd.read_csv(path).sort_values(["target_score"], ascending=False).iloc[0]
    keys = [
        "mom_short",
        "mom_long",
        "threshold",
        "signal_mode",
        "reverse_signal",
        "top_1",
        "top_2",
        "top_3",
        "top_4",
    ]
    out: dict[str, object] = {k: best[k] for k in keys}
    out["mom_short"] = int(out["mom_short"])
    out["mom_long"] = int(out["mom_long"])
    out["threshold"] = int(out["threshold"])
    out["signal_mode"] = str(out["signal_mode"])
    out["reverse_signal"] = bool(out["reverse_signal"])
    out["top_1"] = int(out["top_1"])
    out["top_2"] = int(out["top_2"])
    out["top_3"] = int(out["top_3"])
    out["top_4"] = int(out["top_4"])
    return out


def target_score(metrics: dict[str, float], nonzero_ratio: float) -> float:
    if any(np.isnan(metrics.get(k, np.nan)) for k in ["ann", "sharpe", "calmar", "mdd"]):
        return -np.inf
    ann_gap = abs(metrics["ann"] - TARGET_ANN)
    sharpe_gap = abs(metrics["sharpe"] - TARGET_SHARPE)
    calmar_gap = abs(metrics["calmar"] - TARGET_CALMAR) if pd.notna(metrics["calmar"]) else 10.0
    nonzero_gap = abs(nonzero_ratio - TARGET_NONZERO)
    mdd_penalty = max(abs(metrics["mdd"]) - 0.05, 0.0)
    positive_bonus = 0.4 if metrics["ann"] > 0 and metrics["sharpe"] > 0 else -1.0
    return -(2.0 * ann_gap + 1.5 * sharpe_gap + 1.0 * calmar_gap + 0.5 * nonzero_gap + 2.0 * mdd_penalty) + positive_bonus


@dataclass(frozen=True)
class FixedParams:
    mom_short: int
    mom_long: int
    threshold: int
    reverse_signal: bool
    top_1: int
    top_2: int
    top_3: int
    top_4: int


def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Missing {INPUT_XLSX}. Run build_public_macro_dataset.py first.")

    base = load_second_stage_best_params()
    fixed = FixedParams(
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

    signal_modes = ["or", "and"]
    z_win_levels = [126, 252, 504]
    z_win_diffs = [21, 63, 126]
    z_minp_levels = [60, 90]
    z_minp_diffs = [20]  # must be <= z_win_diff (e.g. 21)
    costs = [0.0, 0.00003, 0.00005]

    rows: list[dict[str, object]] = []
    best_score = float("-inf")
    best_df = None
    best_cfg = None

    total = (
        len(signal_modes)
        * len(z_win_levels)
        * len(z_win_diffs)
        * len(z_minp_levels)
        * len(z_minp_diffs)
        * len(costs)
    )
    print(f"third-stage combos: {total}")

    for i, (signal_mode, z_win_level, z_win_diff, z_minp_level, z_minp_diff, cost) in enumerate(
        product(signal_modes, z_win_levels, z_win_diffs, z_minp_levels, z_minp_diffs, costs),
        start=1,
    ):
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
        nonzero_ratio = float((df["FinalPos"] != 0).mean())
        score = target_score(metrics, nonzero_ratio)

        rows.append(
            {
                "signal_mode": signal_mode,
                "z_win_level": int(z_win_level),
                "z_minp_level": int(z_minp_level),
                "z_win_diff": int(z_win_diff),
                "z_minp_diff": int(z_minp_diff),
                "cost_per_turn": float(cost),
                "mom_short": fixed.mom_short,
                "mom_long": fixed.mom_long,
                "threshold": fixed.threshold,
                "reverse_signal": fixed.reverse_signal,
                "top_1": fixed.top_1,
                "top_2": fixed.top_2,
                "top_3": fixed.top_3,
                "top_4": fixed.top_4,
                "ann": metrics["ann"],
                "vol": metrics["vol"],
                "sharpe": metrics["sharpe"],
                "mdd": metrics["mdd"],
                "calmar": metrics["calmar"],
                "nonzero_ratio": nonzero_ratio,
                "target_score": score,
            }
        )

        if score > best_score:
            best_score = score
            best_df = df.copy()
            best_cfg = cfg

        if i % 10 == 0:
            print(f"processed {i}/{total}; current best score={best_score:.4f}")

    out = pd.DataFrame(rows).sort_values(["target_score", "sharpe", "ann"], ascending=False)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_RESULTS, index=False, encoding="utf-8-sig")

    if best_df is None or best_cfg is None:
        raise RuntimeError("No third-stage result generated")
    best_df.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    print("saved", OUT_RESULTS)
    print("saved", OUT_BEST)
    print("best third-stage cfg:")
    print(best_cfg)
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
