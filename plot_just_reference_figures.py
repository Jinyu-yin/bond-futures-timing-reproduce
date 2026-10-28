from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def nav_csv(self) -> Path:
        return self.project_root / "just_reference_v2_nav.csv"

    @property
    def out_dir(self) -> Path:
        return self.project_root / "reproduce" / "figures"


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: matplotlib. Run: py -m pip install matplotlib") from e


def calc_drawdown(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0


def calc_metrics(ret: pd.Series) -> dict[str, float]:
    r = ret.dropna()
    if len(r) == 0:
        return {"ann": np.nan, "vol": np.nan, "sharpe": np.nan, "mdd": np.nan, "calmar": np.nan, "hit": np.nan}
    ann = (1 + r.mean()) ** 252 - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 1e-12 else np.nan
    nav = (1 + r).cumprod()
    dd = calc_drawdown(nav)
    mdd = float(dd.min())
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    hit = float((r > 0).mean())
    return {"ann": float(ann), "vol": float(vol), "sharpe": float(sharpe), "mdd": float(mdd), "calmar": float(calmar), "hit": hit}


def load_nav(paths: Paths) -> pd.DataFrame:
    if not paths.nav_csv.exists():
        raise FileNotFoundError(
            f"Missing {paths.nav_csv}. Please run just_reference_reproduction_v2.py first to generate it."
        )
    df = pd.read_csv(paths.nav_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    for c in ["Close", "Ret", "RawSignal", "Position", "StrategyRet", "Nav"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["BenchNav"] = (1.0 + df["Ret"].fillna(0.0)).cumprod()
    df["DD_Strategy"] = calc_drawdown(df["Nav"])
    df["DD_Bench"] = calc_drawdown(df["BenchNav"])
    return df


def plot_nav_and_drawdown(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    ax = axes[0]
    ax.plot(df.index, df["BenchNav"], label="Benchmark (10Y Bond Futures)", linewidth=1.5, alpha=0.9)
    ax.plot(df.index, df["Nav"], label="Strategy (Macro factor assetization)", linewidth=1.8)
    ax.set_title("Net Value (after cost) vs Benchmark")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    ax2 = axes[1]
    ax2.fill_between(df.index, df["DD_Strategy"].values, 0, alpha=0.25, label="Strategy DD")
    ax2.plot(df.index, df["DD_Bench"], alpha=0.8, linewidth=1.0, label="Benchmark DD")
    ax2.set_title("Drawdown")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_signal_and_position(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]})

    ax = axes[0]
    ax.plot(df.index, df["RawSignal"], linewidth=1.0)
    ax.axhline(0.0, linewidth=1.0, color="black", alpha=0.5)
    ax.set_title("Composite Signal (RawSignal)")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    ax2.step(df.index, df["Position"], where="post", linewidth=1.0)
    ax2.axhline(0.0, linewidth=1.0, color="black", alpha=0.5)
    ax2.set_title("Position (after threshold + vol targeting)")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_monthly_returns_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    r = df["StrategyRet"].fillna(0.0)
    m = (1.0 + r).resample("ME").prod() - 1.0
    heat = m.to_frame("ret")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    pivot = heat.pivot(index="year", columns="month", values="ret").sort_index()

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.03, vmax=0.03)
    ax.set_title("Monthly Returns Heatmap (Strategy)")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels([str(i) for i in range(1, 13)])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Return")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    paths = Paths(project_root=Path(__file__).resolve().parents[1])
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_nav(paths)

    plot_nav_and_drawdown(df, paths.out_dir / "fig_01_nav_drawdown.png")
    plot_signal_and_position(df, paths.out_dir / "fig_02_signal_position.png")
    plot_monthly_returns_heatmap(df, paths.out_dir / "fig_03_monthly_heatmap.png")

    m_strat = calc_metrics(df["StrategyRet"])
    m_bench = calc_metrics(df["Ret"])
    metrics = pd.DataFrame([m_strat, m_bench], index=["strategy", "benchmark"])
    metrics.to_csv(paths.out_dir / "metrics.csv", encoding="utf-8-sig")
    print("saved", paths.out_dir / "metrics.csv")
    print(metrics.round(4))


if __name__ == "__main__":
    main()

