from __future__ import annotations

from pathlib import Path

import pandas as pd

from plot_paper_style_figures import calc_metrics


ROOT = Path(__file__).resolve().parents[1]
BASE_NAV = ROOT / "reproduce" / "paper_style_nav_public.csv"
FAST_NAV = ROOT / "reproduce" / "paper_style_nav_public_fast_best.csv"
TARGET_NAV = ROOT / "reproduce" / "paper_style_nav_public_paper_target_best.csv"
SECOND_NAV = ROOT / "reproduce" / "paper_style_nav_public_second_stage_best.csv"
OUT_DIR = ROOT / "reproduce" / "figures"
OUT_FIG = OUT_DIR / "fig_10_public_base_fast_target_second.png"
OUT_METRICS = OUT_DIR / "public_base_fast_target_second_metrics.csv"
OUT_FAST_TOP10 = OUT_DIR / "public_fast_top10.csv"
OUT_TARGET_TOP10 = OUT_DIR / "public_paper_target_top10.csv"
FAST_RESULTS = ROOT / "reproduce" / "public_fast_optimization_results.csv"
TARGET_RESULTS = ROOT / "reproduce" / "public_paper_target_optimization_results.csv"


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing dependency: matplotlib") from e


def load_nav(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_all(df_base: pd.DataFrame, df_fast: pd.DataFrame, df_target: pd.DataFrame, df_second: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_base["Date"], df_base["StrategyNav"], label="base", color="#5b8ff9", linewidth=1.8)
    ax.plot(df_fast["Date"], df_fast["StrategyNav"], label="fast-best", color="#61d9a3", linewidth=2.0)
    ax.plot(df_target["Date"], df_target["StrategyNav"], label="paper-target-best", color="#f08c2b", linewidth=2.2)
    ax.plot(df_second["Date"], df_second["StrategyNav"], label="second-stage-best", color="#9d4edd", linewidth=2.3)
    ax.plot(df_base["Date"], df_base["BenchNav"], label="benchmark", color="#5d7092", linewidth=1.2, alpha=0.8)
    ax.set_title("Public-data optimization: base vs fast vs target vs second-stage")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    df_base = load_nav(BASE_NAV)
    df_fast = load_nav(FAST_NAV)
    df_target = load_nav(TARGET_NAV)
    df_second = load_nav(SECOND_NAV)
    plot_all(df_base, df_fast, df_target, df_second, OUT_FIG)

    metrics = {
        "base": calc_metrics(df_base["StrategyRet"]),
        "fast_best": calc_metrics(df_fast["StrategyRet"]),
        "paper_target_best": calc_metrics(df_target["StrategyRet"]),
        "second_stage_best": calc_metrics(df_second["StrategyRet"]),
        "benchmark": calc_metrics(df_base["Ret"]),
    }
    pd.DataFrame(metrics).T.to_csv(OUT_METRICS, encoding="utf-8-sig")

    if FAST_RESULTS.exists():
        pd.read_csv(FAST_RESULTS).head(10).to_csv(OUT_FAST_TOP10, index=False, encoding="utf-8-sig")
    if TARGET_RESULTS.exists():
        pd.read_csv(TARGET_RESULTS).head(10).to_csv(OUT_TARGET_TOP10, index=False, encoding="utf-8-sig")

    print("saved", OUT_FIG)
    print("saved", OUT_METRICS)
    print(pd.DataFrame(metrics).T.round(4).to_string())


if __name__ == "__main__":
    main()
