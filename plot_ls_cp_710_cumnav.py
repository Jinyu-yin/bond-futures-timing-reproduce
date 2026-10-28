from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"
OUT_DIR = ROOT / "reproduce" / "figures"
OUT_FIG = OUT_DIR / "fig_ls_long710_short_cp_cumnav.png"

COL_LONG = "5_long_gov_510y_csi931018_close"
COL_SHORT = "5_short_cp_csi_close"
COL_LS = "5_ls_gov510y_minus_shortcp_nav"


def level_to_cumnav(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    first = s.dropna().iloc[0] if s.notna().any() else float("nan")
    if not first or pd.isna(first) or first == 0:
        return s * float("nan")
    return s / float(first)


def sample_years(dates: pd.Series) -> float:
    d0 = pd.Timestamp(dates.iloc[0])
    d1 = pd.Timestamp(dates.iloc[-1])
    return max((d1 - d0).days / 365.25, 1e-6)


def cagr_from_nav(nav: pd.Series, years: float) -> float:
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2 or years <= 0:
        return float("nan")
    a, b = float(nav.iloc[0]), float(nav.iloc[-1])
    if a <= 0 or b <= 0:
        return float("nan")
    return (b / a) ** (1.0 / years) - 1.0


def ann_vol_daily(r: pd.Series) -> float:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 10:
        return float("nan")
    return float(r.std() * np.sqrt(252.0))


def main() -> None:
    if not FACTORS_CSV.exists():
        raise FileNotFoundError(f"Missing {FACTORS_CSV}. Run reproduce/build_public_macro_dataset.py first.")

    df = pd.read_csv(FACTORS_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    missing = [c for c in (COL_LONG, COL_SHORT, COL_LS) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in factors CSV: {missing}")

    nav_long = level_to_cumnav(df[COL_LONG])
    nav_short = level_to_cumnav(df[COL_SHORT])
    nav_ls = pd.to_numeric(df[COL_LS], errors="coerce")

    years = sample_years(df["Date"])
    c_long = cagr_from_nav(nav_long, years)
    c_short = cagr_from_nav(nav_short, years)
    c_ls = cagr_from_nav(nav_ls, years)
    rl = pd.to_numeric(df[COL_LONG], errors="coerce").pct_change()
    rs = pd.to_numeric(df[COL_SHORT], errors="coerce").pct_change()
    ls_ret = rl - rs
    vol_ls = ann_vol_daily(ls_ret)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(df["Date"], nav_long, label=f"Long 国债5-10y活跃(931018)  CAGR≈{c_long*100:.2f}%", lw=1.2, color="#1f77b4")
    ax.plot(df["Date"], nav_short, label=f"Short 中证短融(H11014)  CAGR≈{c_short*100:.2f}%", lw=1.2, color="#ff7f0e")
    ax.plot(df["Date"], nav_ls, label=f"LS 利差净值  CAGR≈{c_ls*100:.2f}%", lw=1.4, color="black", alpha=0.85)
    ax.set_title("累计净值（起点=1）｜国债(CSI) vs 短融｜年化见图例")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (start = 1)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    note = (
        f"样本跨度≈{years:.2f}年（按日历日）。\n"
        f"LS 口径：日收益 = 国债活跃券指数日收益 − 短融指数日收益，再累乘；"
        f"近似年化波动≈{vol_ls*100:.2f}%（非费后可交易组合）。\n"
        "长端为 CSI 931018（国债），不再使用中债指数；"
        "黑线为利差累计净值，通常弱于两条纯多头线各自涨幅。"
    )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print("saved", OUT_FIG)
    print(
        f"CAGR long={c_long*100:.3f}%  short={c_short*100:.3f}%  LS={c_ls*100:.3f}%  "
        f"ann_vol(ls_daily)={vol_ls*100:.3f}%  years={years:.3f}"
    )


if __name__ == "__main__":
    main()
