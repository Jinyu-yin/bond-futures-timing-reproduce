"""
国债/短融 连续仓位：信号越强越偏向一侧，权重落在 [w_floor, w_cap]，两侧之和为常数（默认 2.0），
故任一侧最低为 w_floor（默认 0.8），最高 w_cap（默认 1.2），不会出现某一侧为 0 的离散切换。

收益（t 日）：用 t-1 收盘确定的权重 w_{t-1} 乘当日 r_t；滑点按 |Δw_gov|+|Δw_cp| 近似。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_to_zscore(s: pd.Series, win: int = 120, minp: int = 30) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std().replace(0, np.nan)
    z = (s - mu) / (sd + 1e-8)
    return z.clip(-6, 6).fillna(0.0)


def bipartite_weights_from_z(
    z: pd.Series,
    *,
    gamma: float = 1.0,
    w_floor: float = 0.8,
    w_cap: float = 1.2,
) -> tuple[pd.Series, pd.Series]:
    """z 越大越偏国债：w_gov ∈ [w_floor, w_cap]，w_cp = (w_floor+w_cap) - w_gov。"""
    if w_floor <= 0 or w_cap <= w_floor:
        raise ValueError("need 0 < w_floor < w_cap")
    total = float(w_floor + w_cap)
    u = (np.tanh(float(gamma) * z) + 1.0) * 0.5
    u = pd.Series(u, index=z.index).clip(0.0, 1.0)
    w_gov = w_floor + (w_cap - w_floor) * u
    w_cp = total - w_gov
    return w_gov.astype(float), w_cp.astype(float)


def daily_strategy_return_continuous(
    r_gov: pd.Series,
    r_cp: pd.Series,
    w_gov: pd.Series,
    w_cp: pd.Series,
    slip_per_unit_turnover: float = 0.0001,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    返回 (strat_ret, w_gov_exec, w_cp_exec)，其中 exe 为 shift(1) 后的执行权重。
    """
    rg = pd.to_numeric(r_gov, errors="coerce").fillna(0.0)
    rs = pd.to_numeric(r_cp, errors="coerce").fillna(0.0)
    wg = pd.to_numeric(w_gov, errors="coerce").fillna(0.0)
    wc = pd.to_numeric(w_cp, errors="coerce").fillna(0.0)

    wg_e = wg.shift(1).fillna(0.0)
    wc_e = wc.shift(1).fillna(0.0)
    strat = wg_e * rg + wc_e * rs
    turn = wg.diff().abs().fillna(0.0) + wc.diff().abs().fillna(0.0)
    strat = strat - float(slip_per_unit_turnover) * turn
    return strat, wg_e, wc_e


def timing_win_rates_tilt(
    r_gov: pd.Series,
    r_cp: pd.Series,
    w_gov_exe: pd.Series,
    w_cp_exe: pd.Series,
) -> dict[str, float]:
    """执行权重下：偏国债日看 r_gov>0；偏短融日看 r_cp>0。"""
    rg = pd.to_numeric(r_gov, errors="coerce").fillna(0.0)
    rs = pd.to_numeric(r_cp, errors="coerce").fillna(0.0)
    wg = pd.to_numeric(w_gov_exe, errors="coerce").fillna(0.0)
    wc = pd.to_numeric(w_cp_exe, errors="coerce").fillna(0.0)
    tilt_g = wg > wc
    tilt_c = wc > wg
    long_wr = float((rg[tilt_g] > 0).mean()) if bool(tilt_g.any()) else float("nan")
    short_wr = float((rs[tilt_c] > 0).mean()) if bool(tilt_c.any()) else float("nan")
    inv = tilt_g | tilt_c
    hit = ((tilt_g & (rg > 0)) | (tilt_c & (rs > 0)))
    inv_wr = float(hit[inv].mean()) if bool(inv.any()) else float("nan")
    return {
        "timing_n_long_days": float(tilt_g.sum()),
        "timing_n_short_days": float(tilt_c.sum()),
        "timing_long_win_rate": long_wr,
        "timing_short_win_rate": short_wr,
        "timing_invested_hit_rate": inv_wr,
    }
