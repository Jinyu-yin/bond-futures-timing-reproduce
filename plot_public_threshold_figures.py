from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plot_paper_style_figures import calc_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reproduce" / "figures"
NAV_0 = ROOT / "reproduce" / "paper_style_nav_public.csv"
NAV_1 = ROOT / "reproduce" / "paper_style_nav_public_th1.csv"
OUT_FIG_THRESHOLD = OUT_DIR / "fig_06_public_threshold_compare.png"
OUT_FIG_HEAT = OUT_DIR / "fig_07_public_monthly_heatmap.png"
OUT_FIG_CORR = OUT_DIR / "fig_08_public_group_corr.png"
OUT_METRICS = OUT_DIR / "public_threshold_compare_metrics.csv"


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib. Run: py -m pip install matplotlib") from e


def load_nav(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_threshold_compare(df0: pd.DataFrame, df1: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df0["Date"], df0["StrategyNav"], label="composite threshold=0", color="#5b8ff9", linewidth=2.2)
    ax.plot(df1["Date"], df1["StrategyNav"], label="composite threshold=1", color="#f6bd16", linewidth=2.2)
    ax.plot(df0["Date"], df0["BenchNav"], label="benchmark", color="#5d7092", linewidth=1.6, alpha=0.85)
    ax.set_title("Public-data composite timing: threshold comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_monthly_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    monthly = df.set_index("Date")["StrategyRet"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    table = monthly.to_frame("ret")
    table["Year"] = table.index.year
    table["Month"] = table.index.month
    pivot = table.pivot(index="Year", columns="Month", values="ret").sort_index()

    fig, ax = plt.subplots(figsize=(12, 5.6))
    vals = pivot.values
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 0.05
    vmax = max(vmax, 0.03)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_title("Public-data composite monthly returns heatmap (threshold=1)")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("monthly return")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_group_corr(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    group_cols = ["1_growth_nav", "2_inflation_nav", "3_credit_nav", "4_omo_nav"]
    ret = df.set_index("Date")[group_cols].pct_change().resample("ME").apply(lambda x: (1 + x).prod() - 1)
    corr = ret.corr()
    labels = ["growth", "inflation", "credit", "omo"]

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Monthly return correlation across factor buckets")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10, color="black")

    fig.colorbar(im, ax=ax, shrink=0.88)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_metrics(df0: pd.DataFrame, df1: pd.DataFrame, out_path: Path) -> None:
    metrics = {
        "threshold_0": calc_metrics(df0["StrategyRet"]),
        "threshold_1": calc_metrics(df1["StrategyRet"]),
        "benchmark": calc_metrics(df0["Ret"]),
    }
    extra = pd.DataFrame(
        {
            "nonzero_ratio": [
                float((df0["FinalPos"] != 0).mean()),
                float((df1["FinalPos"] != 0).mean()),
                1.0,
            ]
        },
        index=["threshold_0", "threshold_1", "benchmark"],
    )
    out = pd.DataFrame(metrics).T.join(extra)
    out.to_csv(out_path, encoding="utf-8-sig")


def main() -> None:
    df0 = load_nav(NAV_0)
    df1 = load_nav(NAV_1)
    plot_threshold_compare(df0, df1, OUT_FIG_THRESHOLD)
    plot_monthly_heatmap(df1, OUT_FIG_HEAT)
    plot_group_corr(df1, OUT_FIG_CORR)
    save_metrics(df0, df1, OUT_METRICS)
    print("saved", OUT_FIG_THRESHOLD)
    print("saved", OUT_FIG_HEAT)
    print("saved", OUT_FIG_CORR)
    print("saved", OUT_METRICS)
    print(pd.read_csv(OUT_METRICS, index_col=0).round(4).to_string())


if __name__ == "__main__":
    main()
