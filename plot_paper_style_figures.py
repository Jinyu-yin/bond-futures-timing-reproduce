from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib. Run: py -m pip install matplotlib") from e


def calc_metrics(ret: pd.Series) -> dict[str, float]:
    r = ret.dropna()
    if len(r) == 0:
        return {"ann": np.nan, "vol": np.nan, "sharpe": np.nan, "mdd": np.nan, "calmar": np.nan}
    ann = (1 + r.mean()) ** 252 - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 1e-12 else np.nan
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    mdd = float(dd.min())
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return {"ann": float(ann), "vol": float(vol), "sharpe": float(sharpe), "mdd": float(mdd), "calmar": float(calmar)}


def plot_factor_category_navs(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    idx = df["Date"] if "Date" in df.columns else df.index

    series = {
        "1_growth": df["1_growth_nav"],
        "2_inflation": df["2_inflation_nav"],
        "3_credit": df["3_credit_nav"],
        "4_omo": df["4_omo_nav"],
        "4-factor composite": df["StrategyNav"],
        "benchmark (buy&hold)": df["BenchNav"],
    }
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#ffbd2e", "#9467bd", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for (name, s), c in zip(series.items(), colors):
        ax.plot(
            idx,
            s.values,
            label=name,
            linewidth=1.6 if name != "4-factor composite" else 2.5,
            color=c,
        )

    ax.set_title("Cumulative NAV by factor bucket + composite (paper-style repro)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    nav_csv = root / "reproduce" / "paper_style_nav.csv"
    out_dir = root / "reproduce" / "figures"
    if not nav_csv.exists():
        raise FileNotFoundError(f"Missing {nav_csv}. Run reproduce/paper_style_bond_futures_timing.py first.")

    df = pd.read_csv(nav_csv)
    df["Date"] = pd.to_datetime(df["Date"])

    plot_factor_category_navs(df, out_dir / "fig_04_paper_style_factor_navs.png")

    m = {
        "composite": calc_metrics(df["StrategyRet"]),
        "benchmark": calc_metrics(df["Ret"]),
    }
    pd.DataFrame(m).T.to_csv(out_dir / "paper_style_metrics.csv", encoding="utf-8-sig")
    print("saved", out_dir / "fig_04_paper_style_factor_navs.png")
    print("saved", out_dir / "paper_style_metrics.csv")
    print(pd.DataFrame(m).T.round(4))


if __name__ == "__main__":
    main()
