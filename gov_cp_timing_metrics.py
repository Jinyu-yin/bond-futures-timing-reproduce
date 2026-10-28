"""
国债/短融切换策略的「择时」胜率（不含票息拆分，仅按标的日收益统计）。

- 多头胜率：上一交易日持有国债时，当日国债收益率 > 0 的占比。
- 空头胜率：上一交易日持有短融时，当日短融收益率 > 0 的占比（短融端亦为指数多头持仓）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def timing_win_rates_gov_cp(
    r_gov: pd.Series,
    r_cp: pd.Series,
    w_gov: pd.Series,
    w_cp: pd.Series,
) -> dict[str, float]:
    """
    w_gov, w_cp: 当日收盘确定的持仓权重（0/1），与 build_public_macro_dataset 一致；
    押后一日执行：用 shift(1) 对齐到收益日。
    """
    rg = pd.to_numeric(r_gov, errors="coerce").fillna(0.0)
    rc = pd.to_numeric(r_cp, errors="coerce").fillna(0.0)
    wg = pd.to_numeric(w_gov, errors="coerce").fillna(0.0)
    wc = pd.to_numeric(w_cp, errors="coerce").fillna(0.0)

    prev_g = wg.shift(1).fillna(0.0) > 0.5
    prev_c = wc.shift(1).fillna(0.0) > 0.5

    long_wr = float((rg[prev_g] > 0).mean()) if bool(prev_g.any()) else float("nan")
    short_wr = float((rc[prev_c] > 0).mean()) if bool(prev_c.any()) else float("nan")

    invested = prev_g | prev_c
    strat_r = prev_g.astype(float) * rg + prev_c.astype(float) * rc
    hit_inv = float((strat_r[invested] > 0).mean()) if bool(invested.any()) else float("nan")

    return {
        "timing_n_long_days": float(prev_g.sum()),
        "timing_n_short_days": float(prev_c.sum()),
        "timing_long_win_rate": long_wr,
        "timing_short_win_rate": short_wr,
        "timing_invested_hit_rate": hit_inv,
    }


def timing_win_rates_from_position(
    r_gov: pd.Series,
    r_cp: pd.Series,
    pos: pd.Series,
) -> dict[str, float]:
    """pos: +1 国债, -1 短融, 0 空仓；与 v2 离散仓位一致。"""
    p = pd.to_numeric(pos, errors="coerce").fillna(0.0)
    w_gov = (p > 0).astype(float)
    w_cp = (p < 0).astype(float)
    return timing_win_rates_gov_cp(r_gov, r_cp, w_gov, w_cp)


def longonly_buyhold_win_rate(r_gov: pd.Series) -> dict[str, float]:
    """买入持有国债：仅统计『多头』胜率 = 国债上涨日占比。"""
    rg = pd.to_numeric(r_gov, errors="coerce").dropna()
    if rg.empty:
        return {
            "timing_n_long_days": 0.0,
            "timing_n_short_days": 0.0,
            "timing_long_win_rate": float("nan"),
            "timing_short_win_rate": float("nan"),
            "timing_invested_hit_rate": float("nan"),
        }
    hit = float((rg > 0).mean())
    return {
        "timing_n_long_days": float(len(rg)),
        "timing_n_short_days": 0.0,
        "timing_long_win_rate": hit,
        "timing_short_win_rate": float("nan"),
        "timing_invested_hit_rate": hit,
    }


def fmt_pct(x: float, digits: int = 2) -> str:
    if x != x or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{digits}f}%"
