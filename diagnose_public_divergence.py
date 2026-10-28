from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_NAV = ROOT / "reproduce" / "paper_style_nav_public.csv"
BEST_NAV = ROOT / "reproduce" / "paper_style_nav_public_best.csv"
OUT_CSV = ROOT / "reproduce" / "public_divergence_diagnosis.csv"


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def monthly_return(series: pd.Series) -> pd.Series:
    return series.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def top_drawdown_dates(dd: pd.Series, n: int = 10) -> pd.DataFrame:
    worst = dd.nsmallest(n)
    return pd.DataFrame({"date": worst.index, "drawdown": worst.values})


def main() -> None:
    if not BASE_NAV.exists() or not BEST_NAV.exists():
        raise FileNotFoundError("Run optimize_public_paper_style.py first.")

    base = pd.read_csv(BASE_NAV)
    best = pd.read_csv(BEST_NAV)
    base["Date"] = pd.to_datetime(base["Date"])
    best["Date"] = pd.to_datetime(best["Date"])
    base = base.set_index("Date")
    best = best.set_index("Date")

    rows: list[dict[str, object]] = []

    for label, df in [("base", base), ("optimized", best)]:
        rows.append(
            {
                "section": "position_summary",
                "strategy": label,
                "metric": "nonzero_ratio",
                "value": float((df["FinalPos"] != 0).mean()),
            }
        )
        rows.append(
            {
                "section": "position_summary",
                "strategy": label,
                "metric": "long_ratio",
                "value": float((df["FinalPos"] > 0).mean()),
            }
        )
        rows.append(
            {
                "section": "position_summary",
                "strategy": label,
                "metric": "short_ratio",
                "value": float((df["FinalPos"] < 0).mean()),
            }
        )

        for col in ["1_growth_nav", "2_inflation_nav", "3_credit_nav", "4_omo_nav"]:
            ann = (df[col].iloc[-1] ** (252 / max(len(df), 1))) - 1
            rows.append(
                {
                    "section": "group_nav_terminal",
                    "strategy": label,
                    "metric": col,
                    "value": float(df[col].iloc[-1]),
                    "annualized_proxy": float(ann),
                }
            )

        mret = monthly_return(df["StrategyRet"])
        rows.append(
            {
                "section": "monthly_summary",
                "strategy": label,
                "metric": "positive_month_ratio",
                "value": float((mret > 0).mean()),
            }
        )
        rows.append(
            {
                "section": "monthly_summary",
                "strategy": label,
                "metric": "avg_monthly_return",
                "value": float(mret.mean()),
            }
        )

        dd = drawdown(df["StrategyNav"])
        worst = top_drawdown_dates(dd, n=8)
        for _, row in worst.iterrows():
            rows.append(
                {
                    "section": "worst_drawdowns",
                    "strategy": label,
                    "metric": str(pd.Timestamp(row["date"]).date()),
                    "value": float(row["drawdown"]),
                }
            )

    base_month = monthly_return(base["StrategyRet"])
    best_month = monthly_return(best["StrategyRet"])
    aligned = pd.concat([base_month.rename("base"), best_month.rename("optimized")], axis=1).dropna()
    rows.append(
        {
            "section": "comparison",
            "strategy": "base_vs_optimized",
            "metric": "monthly_return_corr",
            "value": float(aligned["base"].corr(aligned["optimized"])),
        }
    )
    rows.append(
        {
            "section": "comparison",
            "strategy": "base_vs_optimized",
            "metric": "optimized_minus_base_terminal_nav",
            "value": float(best["StrategyNav"].iloc[-1] - base["StrategyNav"].iloc[-1]),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("saved", OUT_CSV)
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
