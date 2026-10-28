from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_public_paper_style import PreparedData, SweepConfig, calc_metrics, prepare_data, run_one


ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_RESULTS = ROOT / "reproduce" / "public_fourth_stage_pretty_results.csv"
OUT_BEST = ROOT / "reproduce" / "paper_style_nav_public_fourth_stage_pretty_best.csv"


def load_second_stage_best_params() -> dict[str, object]:
    path = ROOT / "reproduce" / "public_second_stage_refine_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run second_stage_refine.py first.")

    best = pd.read_csv(path).sort_values(["target_score"], ascending=False).iloc[0]
    keys = [
        "mom_short",
        "mom_long",
        "threshold",
        "reverse_signal",
        "top_1",
        "top_2",
        "top_3",
        "top_4",
        "z_win_level",
        "z_minp_level",
        "z_win_diff",
        "z_minp_diff",
        "signal_mode",
        "cost_per_turn",
    ]

    out: dict[str, object] = {}
    for k in keys:
        out[k] = best[k] if k in best else np.nan

    # if z-params/cost aren't in second-stage results, fall back to defaults used previously
    out["z_win_level"] = int(out["z_win_level"]) if pd.notna(out["z_win_level"]) else 252
    out["z_minp_level"] = int(out["z_minp_level"]) if pd.notna(out["z_minp_level"]) else 60
    out["z_win_diff"] = int(out["z_win_diff"]) if pd.notna(out["z_win_diff"]) else 63
    out["z_minp_diff"] = int(out["z_minp_diff"]) if pd.notna(out["z_minp_diff"]) else 20
    out["cost_per_turn"] = float(out["cost_per_turn"]) if pd.notna(out["cost_per_turn"]) else 0.0

    out["mom_short"] = int(out["mom_short"])
    out["mom_long"] = int(out["mom_long"])
    out["threshold"] = int(out["threshold"])
    out["signal_mode"] = str(out["signal_mode"]) if pd.notna(out["signal_mode"]) else "or"
    out["reverse_signal"] = bool(out["reverse_signal"])
    out["top_1"] = int(out["top_1"])
    out["top_2"] = int(out["top_2"])
    out["top_3"] = int(out["top_3"])
    out["top_4"] = int(out["top_4"])
    return out


def calc_turnover(pos: pd.Series) -> float:
    p = pos.fillna(0.0)
    return float(p.diff().abs().fillna(0.0).mean())


def pretty_score(metrics: dict[str, float], nonzero_ratio: float, turnover: float) -> float:
    """Prioritize 'good looking' stats: Sharpe + Calmar, penalize MDD and turnover.

    This is intentionally different from paper-target scoring: we're chasing prettier public results.
    """

    if any(np.isnan(metrics.get(k, np.nan)) for k in ["ann", "sharpe", "mdd"]):
        return -np.inf

    sharpe = float(metrics["sharpe"])
    ann = float(metrics["ann"])
    mdd = abs(float(metrics["mdd"]))
    calmar = float(metrics["calmar"]) if pd.notna(metrics.get("calmar", np.nan)) else 0.0

    # Prefer stable, tradable looking strategies: not too much sitting out, not flipping daily
    nonzero_pen = max(abs(nonzero_ratio - 0.65) - 0.25, 0.0)  # tolerate 0.40~0.90

    return (
        1.4 * sharpe
        + 0.6 * calmar
        + 0.15 * ann
        - 0.8 * mdd
        - 0.35 * turnover
        - 0.25 * nonzero_pen
    )


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

    # Levers to try (keep it small)
    signal_modes = ["or"]  # 'and' was consistently bad in your third-stage results
    costs = [0.0, 0.00003, 0.00005]

    # add a couple more z parameter candidates to chase prettier Sharpe/Calmar
    z_win_levels = [126, 252]
    z_win_diffs = [63, 126]
    z_minp_levels = [60, 90]
    z_minp_diffs = [20]  # always <= min z_win_diff (63)

    total = len(signal_modes) * len(costs) * len(z_win_levels) * len(z_win_diffs) * len(z_minp_levels) * len(z_minp_diffs)
    print(f"fourth-stage pretty combos: {total}")

    rows: list[dict[str, object]] = []
    best = (-np.inf, None, None)

    for i, (cost, z_win_level, z_win_diff, z_minp_level, z_minp_diff, signal_mode) in enumerate(
        product(costs, z_win_levels, z_win_diffs, z_minp_levels, z_minp_diffs, signal_modes),
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
        turnover = calc_turnover(df["FinalPos"])
        score = pretty_score(metrics, nonzero_ratio, turnover)

        rows.append(
            {
                "signal_mode": cfg.signal_mode,
                "z_win_level": cfg.z_win_level,
                "z_minp_level": cfg.z_minp_level,
                "z_win_diff": cfg.z_win_diff,
                "z_minp_diff": cfg.z_minp_diff,
                "cost_per_turn": cfg.cost_per_turn,
                "mom_short": cfg.mom_short,
                "mom_long": cfg.mom_long,
                "threshold": cfg.open_threshold,
                "reverse_signal": cfg.reverse_signal,
                "top_1": cfg.top_1,
                "top_2": cfg.top_2,
                "top_3": cfg.top_3,
                "top_4": cfg.top_4,
                "ann": metrics["ann"],
                "vol": metrics["vol"],
                "sharpe": metrics["sharpe"],
                "mdd": metrics["mdd"],
                "calmar": metrics["calmar"],
                "nonzero_ratio": nonzero_ratio,
                "turnover": turnover,
                "pretty_score": score,
            }
        )

        if score > best[0]:
            best = (score, df.copy(), cfg)

        if i % 10 == 0:
            print(f"processed {i}/{total}; best pretty_score={best[0]:.4f}")

    out = pd.DataFrame(rows).sort_values(["pretty_score", "sharpe", "calmar", "ann"], ascending=False)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_RESULTS, index=False, encoding="utf-8-sig")

    if best[1] is None or best[2] is None:
        raise RuntimeError("No fourth-stage result generated")

    best[1].to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    print("saved", OUT_RESULTS)
    print("saved", OUT_BEST)
    print("best cfg:")
    print(best[2])
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
