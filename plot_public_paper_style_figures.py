from __future__ import annotations

from pathlib import Path

import pandas as pd

from plot_paper_style_figures import calc_metrics, plot_factor_category_navs


ROOT = Path(__file__).resolve().parents[1]
NAV_CSV = ROOT / "reproduce" / "paper_style_nav_public.csv"
OUT_DIR = ROOT / "reproduce" / "figures"
OUT_FIG = OUT_DIR / "fig_05_public_data_factor_navs.png"
OUT_METRICS = OUT_DIR / "public_data_metrics.csv"


def main() -> None:
    if not NAV_CSV.exists():
        raise FileNotFoundError(f"Missing {NAV_CSV}. Run run_public_paper_style.py first.")

    df = pd.read_csv(NAV_CSV)
    df["Date"] = pd.to_datetime(df["Date"])

    plot_factor_category_navs(df, OUT_FIG)

    metrics = {
        "public_composite": calc_metrics(df["StrategyRet"]),
        "public_benchmark": calc_metrics(df["Ret"]),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).T.to_csv(OUT_METRICS, encoding="utf-8-sig")
    print("saved", OUT_FIG)
    print("saved", OUT_METRICS)
    print(pd.DataFrame(metrics).T.round(4))


if __name__ == "__main__":
    main()
