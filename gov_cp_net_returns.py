"""中债指数净价 → 按交易日历对齐的日收益率（用于 gov/cp 报告变体）。"""

from __future__ import annotations

import pandas as pd


def fetch_cbond_net_level_series(period: str) -> pd.Series:
    import akshare as ak

    df = ak.bond_composite_index_cbond(indicator="净价", period=period)
    dt = pd.to_datetime(df["date"])
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=dt).sort_index()
    return s


def daily_returns_from_levels_on_calendar(levels: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    lv = levels.reindex(levels.index.union(cal)).sort_index().ffill()
    lv = lv.reindex(cal).ffill().bfill()
    return lv.pct_change().fillna(0.0)


def gov_cp_net_daily_returns_aligned(calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """
    国债腿：中债综合指数净价，待偿期 7–10 年；
    短端腿：中债净价「1年以下」作短久期代理（与 CSI 短融非同一指数，仅保证双边均为净价口径）。
    """
    lg = fetch_cbond_net_level_series("7-10年")
    sh = fetch_cbond_net_level_series("1年以下")
    cal = pd.DatetimeIndex(calendar)
    return daily_returns_from_levels_on_calendar(lg, cal), daily_returns_from_levels_on_calendar(sh, cal)
