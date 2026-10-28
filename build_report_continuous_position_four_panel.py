"""
连续仓位报告：V1 / V2 的 LongShort 均改为「信号强度 → 国债/短融权重连续映射」，
任一侧权重 ∈ [0.8, 1.2] 且和为 2.0（看多国债最高 1.2、看空时国债仍 ≥0.8，不会清空一侧）。

- V1 信号：利差日收益 (r_gov−r_cp) 的滚动 Z
- V2 信号：Dynamic 合成值的滚动 Z（与离散版同源因子筛选）

输出：reproduce/reports/gov_cp_ls/report_continuous_position/

运行：py reproduce/build_report_continuous_position_four_panel.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_ls_gov_cp_report_v2 import compute_v2_daily_continuous_panel
from gov_cp_four_panel_kit import (
    COL_LONG,
    COL_SHORT,
    build_metrics_rows,
    timing_win_rates_tilt,
    v1_weekly_navs_from_factors_csv,
    v1_continuous_daily_panel,
    v2_weekly_navs_from_daily,
    write_four_panel_variant_report,
)
from gov_cp_timing_metrics import longonly_buyhold_win_rate

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"
OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls" / "report_continuous_position"

# 连续映射参数（可按需微调）
CONT_GAMMA = 1.0
W_FLOOR = 0.8
W_CAP = 1.2
V1_DIFF_MEAN_WIN = 20
V1_DIFF_SD_WIN = 60
V2_Z_WIN = 120
V2_Z_MINP = 30


def main() -> None:
    base_daily = pd.read_csv(FACTORS_CSV)
    base_daily["Date"] = pd.to_datetime(base_daily["Date"])
    base_daily = base_daily.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    cal = pd.DatetimeIndex(base_daily["Date"])
    rl_d = pd.Series(
        pd.to_numeric(base_daily[COL_LONG], errors="coerce").pct_change().fillna(0.0).values,
        index=cal,
    )
    rs_d = pd.Series(
        pd.to_numeric(base_daily[COL_SHORT], errors="coerce").pct_change().fillna(0.0).values,
        index=cal,
    )

    daily_v1 = v1_continuous_daily_panel(
        base_daily,
        rl_d,
        rs_d,
        cont_gamma=CONT_GAMMA,
        w_floor=W_FLOOR,
        w_cap=W_CAP,
        diff_mean_win=V1_DIFF_MEAN_WIN,
        diff_sd_win=V1_DIFF_SD_WIN,
        slip=0.0001,
        long_weight=1.0,
        short_weight=1.0,
    )
    daily_v2, _, _, turn_v2 = compute_v2_daily_continuous_panel(
        cont_gamma=CONT_GAMMA,
        w_floor=W_FLOOR,
        w_cap=W_CAP,
        z_win=V2_Z_WIN,
        z_minp=V2_Z_MINP,
    )

    nav_lo_v1, _ = v1_weekly_navs_from_factors_csv(base_daily)
    nav_lo_v2, _ = v1_weekly_navs_from_factors_csv(base_daily)
    _, nav_ls_v1 = v2_weekly_navs_from_daily(daily_v1)
    _, nav_ls_v2 = v2_weekly_navs_from_daily(daily_v2)

    idx = (
        nav_lo_v1.index.intersection(nav_ls_v1.index)
        .intersection(nav_lo_v2.index)
        .intersection(nav_ls_v2.index)
        .sort_values()
    )
    if len(idx) < 5:
        raise RuntimeError("周频交集过短")

    def norm_slice(nav: pd.Series) -> pd.Series:
        s = nav.reindex(idx).ffill()
        a0 = float(s.iloc[0])
        return s / a0 if a0 else s

    s_v1_lo = norm_slice(nav_lo_v1)
    s_v2_lo = norm_slice(nav_lo_v2)
    s_v1_ls = norm_slice(nav_ls_v1)
    s_v2_ls = norm_slice(nav_ls_v2)

    tw_lo = longonly_buyhold_win_rate(rl_d.reindex(cal).fillna(0.0))
    tw_v1_ls = timing_win_rates_tilt(
        daily_v1["r_gov"], daily_v1["r_cp"], daily_v1["w_gov_exe"], daily_v1["w_cp_exe"]
    )
    tw_v2_ls = timing_win_rates_tilt(
        daily_v2["r_gov"], daily_v2["r_cp"], daily_v2["w_gov_exe"], daily_v2["w_cp_exe"]
    )

    rows = build_metrics_rows(
        idx=idx,
        s_v1_lo=s_v1_lo,
        s_v1_ls=s_v1_ls,
        s_v2_lo=s_v2_lo,
        s_v2_ls=s_v2_ls,
        tw_lo=tw_lo,
        tw_v1_ls=tw_v1_ls,
        tw_v2_ls=tw_v2_ls,
    )
    rows[1]["tag"] = "V1 LongShort（连续）"
    rows[3]["tag"] = "V2 LongShort（连续）"

    panels = [
        (s_v1_lo, "V1 · LongOnly", "CSI 931018 买入持有", "#1f77b4"),
        (s_v2_lo, "V2 · LongOnly", "CSI 931018 买入持有", "#9467bd"),
        (s_v1_ls, "V1 · LongShort（连续）", f"利差 Z→权重[{W_FLOOR},{W_CAP}]", "#ff7f0e"),
        (s_v2_ls, "V2 · LongShort（连续）", f"Dynamic Z→权重[{W_FLOOR},{W_CAP}]", "#2ca02c"),
    ]

    note = (
        "<h2>连续仓位规则</h2>"
        f"<p>权重：w<sub>国债</sub> = {W_FLOOR} + ({W_CAP}−{W_FLOOR})·(1+tanh(γ·z))/2，w<sub>短融</sub> = {W_FLOOR + W_CAP} − w<sub>国债</sub>，"
        f"γ={CONT_GAMMA}。任一侧 ∈ [{W_FLOOR}, {W_CAP}]，不会出现 0/1 硬切换。</p>"
        "<p>V1：z 为 (r<sub>国债</sub>−r<sub>短融</sub>) 的滚动均值/波动率；V2：z 为 Dynamic 的滚动标准化（窗口 "
        f"{V2_Z_WIN}/{V2_Z_MINP}）。滑点按 |Δw<sub>国债</sub>|+|Δw<sub>短融</sub>| 扣减。</p>"
        "<p>择时胜率：偏国债日（执行权重 w<sub>国债</sub>&gt;w<sub>短融</sub>）计国债日收益&gt;0；偏短融日对称。</p>"
    )

    write_four_panel_variant_report(
        OUT_DIR,
        idx=idx,
        panels=panels,
        rows_metrics=rows,
        turn_v2=turn_v2,
        report_h1="连续仓位：V1 / V2 × LongOnly / LongShort（四面板）",
        suptitle=f"连续权重 [{W_FLOOR}, {W_CAP}]｜V1 利差 Z · V2 Dynamic Z",
        html_note_p=note,
    )


if __name__ == "__main__":
    main()
