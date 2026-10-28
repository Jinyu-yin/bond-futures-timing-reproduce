from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from optimize_public_paper_style import SweepConfig, calc_metrics, prepare_data, run_one
from optimize_public_paper_target import target_score


ROOT = Path(__file__).resolve().parents[1]
RANKED = ROOT / "reproduce" / "public_ranked_most_paper_like.csv"
OUT_RESULTS = ROOT / "reproduce" / "public_second_stage_refine_results.csv"
OUT_BEST = ROOT / "reproduce" / "paper_style_nav_public_second_stage_best.csv"


def main() -> None:
    if not RANKED.exists():
        raise FileNotFoundError(f"Missing {RANKED}. Run rank_most_paper_like.py first.")

    # speed target: ~20 minutes on typical laptop
    ranked = pd.read_csv(RANKED).head(2)
    if ranked.empty:
        raise RuntimeError("No ranked parameter rows available")

    input_xlsx = ROOT / "reproduce" / "public_input_data.xlsx"
    data = prepare_data(SweepConfig(input_xlsx=input_xlsx))

    rows = []
    best_score = float("-inf")
    best_df = None
    best_cfg = None

    def around_int(x: int, steps: list[int], lo: int, hi: int) -> list[int]:
        vals = {x}
        for s in steps:
            vals.add(max(lo, min(hi, x + s)))
        return sorted(vals)

    for base_i, (_, base) in enumerate(ranked.iterrows(), start=1):
        base_mom_short = int(base["mom_short"])
        base_mom_long = int(base["mom_long"])

        # very local neighborhood
        mom_short_list = sorted(set([base_mom_short] + [v for v in [5, 10, 15, 20] if abs(v - base_mom_short) <= 5]))
        mom_long_list = sorted(set([base_mom_long] + [v for v in [126, 189, 252] if abs(v - base_mom_long) <= 63]))

        top1_list = around_int(int(base["top_1"]), [-2, 0, 2], lo=3, hi=20)
        top2_list = around_int(int(base["top_2"]), [-2, 0, 2], lo=3, hi=20)
        top3_list = around_int(int(base["top_3"]), [-2, 0, 2], lo=2, hi=15)
        top4_list = around_int(int(base["top_4"]), [-1, 0, 1], lo=1, hi=10)

        # keep discrete switches fixed (big speed win)
        threshold_list = [int(base["threshold"])]
        signal_mode_list = [str(base["signal_mode"])]
        reverse_list = [bool(base["reverse_signal"])]

        base_cost = float(base.get("cost_per_turn", 0.00005))
        cost_list = sorted(set([base_cost, 0.00003, 0.00005]))

        combos = (
            len(mom_short_list)
            * len(mom_long_list)
            * len(threshold_list)
            * len(signal_mode_list)
            * len(reverse_list)
            * len(top1_list)
            * len(top2_list)
            * len(top3_list)
            * len(top4_list)
            * len(cost_list)
        )
        print(f"refining base#{base_i}: ~{combos} combos")

        tested = 0
        for mom_short, mom_long, threshold, signal_mode, reverse_signal, top1, top2, top3, top4, cost in product(
            mom_short_list,
            mom_long_list,
            threshold_list,
            signal_mode_list,
            reverse_list,
            top1_list,
            top2_list,
            top3_list,
            top4_list,
            cost_list,
        ):
            if mom_short >= mom_long:
                continue

            cfg = SweepConfig(
                input_xlsx=input_xlsx,
                mom_short=int(mom_short),
                mom_long=int(mom_long),
                open_threshold=int(threshold),
                signal_mode=str(signal_mode),
                reverse_signal=bool(reverse_signal),
                top_1=int(top1),
                top_2=int(top2),
                top_3=int(top3),
                top_4=int(top4),
                cost_per_turn=float(cost),
            )
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
                }
            )
            tested += 1
            if score > best_score:
                best_score = score
                best_df = df.copy()
                best_cfg = cfg
            if tested % 25 == 0:
                print(f"  tested {tested} combos; current best score={best_score:.4f}")

    result = pd.DataFrame(rows).drop_duplicates().sort_values(["target_score", "ann", "sharpe"], ascending=False)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_RESULTS, index=False, encoding="utf-8-sig")
    if best_df is None or best_cfg is None:
        raise RuntimeError("No second-stage result generated")
    best_df.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    print("saved", OUT_RESULTS)
    print("saved", OUT_BEST)
    print(best_cfg)
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
