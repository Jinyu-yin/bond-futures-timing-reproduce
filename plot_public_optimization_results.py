from __future__ import annotations

from pathlib import Path

import pandas as pd

from plot_paper_style_figures import calc_metrics, plot_factor_category_navs


ROOT = Path(__file__).resolve().parents[1]
BASE_NAV = ROOT / "reproduce" / "paper_style_nav_public.csv"
BEST_NAV = ROOT / "reproduce" / "paper_style_nav_public_best.csv"
OPT_RESULTS = ROOT / "reproduce" / "public_optimization_results.csv"
OUT_DIR = ROOT / "reproduce" / "figures"
OUT_FIG = OUT_DIR / "fig_09_public_best_vs_base.png"
OUT_METRICS = OUT_DIR / "public_best_vs_base_metrics.csv"
OUT_TOP10 = OUT_DIR / "public_optimization_top10.csv"


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


def plot_best_vs_base(df_base: pd.DataFrame, df_best: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_base["Date"], df_base["StrategyNav"], label="public base", color="#5b8ff9", linewidth=2.0)
    ax.plot(df_best["Date"], df_best["StrategyNav"], label="public optimized", color="#f08c2b", linewidth=2.2)
    ax.plot(df_base["Date"], df_base["BenchNav"], label="benchmark", color="#5d7092", linewidth=1.4, alpha=0.8)
    ax.set_title("Public-data strategy: base vs optimized")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    df_base = load_nav(BASE_NAV)
    df_best = load_nav(BEST_NAV)
    plot_best_vs_base(df_base, df_best, OUT_FIG)

    metrics = {
        "base": calc_metrics(df_base["StrategyRet"]),
        "optimized": calc_metrics(df_best["StrategyRet"]),
        "benchmark": calc_metrics(df_base["Ret"]),
    }
    pd.DataFrame(metrics).T.to_csv(OUT_METRICS, encoding="utf-8-sig")

    if OPT_RESULTS.exists():
        top10 = pd.read_csv(OPT_RESULTS).head(10)
        top10.to_csv(OUT_TOP10, index=False, encoding="utf-8-sig")

    print("saved", OUT_FIG)
    print("saved", OUT_METRICS)
    if OPT_RESULTS.exists():
        print("saved", OUT_TOP10)
    print(pd.DataFrame(metrics).T.round(4).to_string())


if __name__ == "__main__":
    main()
