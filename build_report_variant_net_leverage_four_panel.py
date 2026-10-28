"""
变体报告 B：净价指数回测 + V1/V2 名义杠杆（国债腿 ×1.2，短融腿 ×0.8），V2 离散阈值保持默认 0.5。

- 收益序列：中债净价 7–10 年（国债）、中债净价 1 年以下（短端代理），对齐 public_factors 日历
- 切换信号：V1 仍用因子表权重；V2 仍用 input_data 因子 Dynamic，阈值不减仓

输出：reproduce/reports/gov_cp_ls/report_variant_net_leverage_120_80/

运行：py reproduce/build_report_variant_net_leverage_four_panel.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from build_ls_gov_cp_report_v2 import compute_v2_daily_strategy_panel
from gov_cp_four_panel_kit import (
    build_metrics_rows,
    v1_switch_daily_leveraged_panel,
    v2_weekly_navs_from_daily,
    write_four_panel_variant_report,
)
from gov_cp_net_returns import gov_cp_net_daily_returns_aligned
from gov_cp_timing_metrics import longonly_buyhold_win_rate, timing_win_rates_gov_cp, timing_win_rates_from_position

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"
OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls" / "report_variant_net_leverage_120_80"

LONG_W = 1.2
SHORT_W = 0.8


def main() -> None:
    base_daily = pd.read_csv(FACTORS_CSV)
    base_daily["Date"] = pd.to_datetime(base_daily["Date"])
    base_daily = base_daily.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    cal = pd.DatetimeIndex(base_daily["Date"])
    rg_net, rs_net = gov_cp_net_daily_returns_aligned(cal)
    rg_s = pd.Series(rg_net.values, index=cal)
    rs_s = pd.Series(rs_net.values, index=cal)

    daily_v1 = v1_switch_daily_leveraged_panel(base_daily, rg_s, rs_s, LONG_W, SHORT_W)
    nav_lo_v1, nav_ls_v1 = v2_weekly_navs_from_daily(daily_v1)

    daily_v2, _, _, turn_v2 = compute_v2_daily_strategy_panel(
        long_weight=LONG_W,
        short_weight=SHORT_W,
        r_gov_override=rg_s,
        r_cp_override=rs_s,
    )
    nav_lo_v2, nav_ls_v2 = v2_weekly_navs_from_daily(daily_v2)

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
    s_v1_ls = norm_slice(nav_ls_v1)
    s_v2_lo = norm_slice(nav_lo_v2)
    s_v2_ls = norm_slice(nav_ls_v2)

    tw_lo = longonly_buyhold_win_rate(rg_s)
    wg_s = pd.Series(
        pd.to_numeric(base_daily["5_pos_switch_gov_weight"], errors="coerce").fillna(0.0).values,
        index=cal,
    )
    wc_s = pd.Series(
        pd.to_numeric(base_daily["5_pos_switch_cp_weight"], errors="coerce").fillna(0.0).values,
        index=cal,
    )
    tw_v1_ls = timing_win_rates_gov_cp(rg_s, rs_s, wg_s, wc_s)
    tw_v2_ls = timing_win_rates_from_position(daily_v2["r_gov"], daily_v2["r_cp"], daily_v2["Pos"])

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

    panels = [
        (s_v1_lo, "V1 · LongOnly", f"净价收益 ×{LONG_W}（中债7–10y）", "#1f77b4"),
        (s_v2_lo, "V2 · LongOnly", f"净价收益 ×{LONG_W}（与 V1 同源序列）", "#9467bd"),
        (s_v1_ls, "V1 · LongShort", f"利差切换，净价腿 ×{LONG_W}/×{SHORT_W}", "#ff7f0e"),
        (s_v2_ls, "V2 · LongShort", f"因子 Dynamic，阈值±0.5，净价腿 ×{LONG_W}/×{SHORT_W}", "#2ca02c"),
    ]

    note = (
        "<h2>设定</h2>"
        "<p>标的日收益：<b>中债净价</b> 7–10年（国债）、<b>中债净价</b> 1年以下（短端代理，非 CSI 短融）。</p>"
        f"<p>名义杠杆：持有国债日收益 ×{LONG_W}，持有短融日收益 ×{SHORT_W}；滑点口径不变。</p>"
        "<p>择时胜率按<strong>净价日收益</strong>统计（不含杠杆倍数）。</p>"
    )

    write_four_panel_variant_report(
        OUT_DIR,
        idx=idx,
        panels=panels,
        rows_metrics=rows,
        turn_v2=turn_v2,
        report_h1="变体：净价 + 杠杆 120%/80%（四策略分面板）",
        suptitle=f"四策略（净价标的 ×{LONG_W}/×{SHORT_W}）｜V2 阈值未放宽",
        html_note_p=note,
    )


if __name__ == "__main__":
    main()
