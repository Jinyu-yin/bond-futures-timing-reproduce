"""
变体报告 A：放宽 V2 离散出手阈值（仅影响 V2 LongShort；V1 仍为 CSI + 原阈值逻辑）。

- 默认 V2：pos_threshold=0.5 → 本脚本改为 0.35（更容易触发 ±1 仓位）
- 四面板输出目录：reproduce/reports/gov_cp_ls/report_variant_v2_relaxed/

运行：py reproduce/build_report_variant_v2_relaxed_four_panel.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from build_ls_gov_cp_report_v2 import SOPConfig, compute_v2_daily_strategy_panel
from gov_cp_four_panel_kit import (
    COL_LONG,
    COL_SHORT,
    build_metrics_rows,
    v1_weekly_navs_from_factors_csv,
    v2_weekly_navs_from_daily,
    write_four_panel_variant_report,
)
from gov_cp_timing_metrics import longonly_buyhold_win_rate, timing_win_rates_gov_cp, timing_win_rates_from_position

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"
OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls" / "report_variant_v2_relaxed"

V2_RELAXED_POS_THRESHOLD = 0.35


def main() -> None:
    base_daily = pd.read_csv(FACTORS_CSV)
    base_daily["Date"] = pd.to_datetime(base_daily["Date"])
    base_daily = base_daily.sort_values("Date")

    cfg = replace(SOPConfig(), pos_threshold=V2_RELAXED_POS_THRESHOLD)
    daily_v2, _, _, turn_v2 = compute_v2_daily_strategy_panel(cfg=cfg)

    nav_lo_v1, nav_ls_v1 = v1_weekly_navs_from_factors_csv(base_daily)
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

    rl_d = pd.to_numeric(base_daily[COL_LONG], errors="coerce").pct_change()
    rs_d = pd.to_numeric(base_daily[COL_SHORT], errors="coerce").pct_change()
    tw_lo = longonly_buyhold_win_rate(rl_d)
    tw_v1_ls = timing_win_rates_gov_cp(
        rl_d,
        rs_d,
        pd.to_numeric(base_daily["5_pos_switch_gov_weight"], errors="coerce").fillna(0.0),
        pd.to_numeric(base_daily["5_pos_switch_cp_weight"], errors="coerce").fillna(0.0),
    )
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

    th = cfg.pos_threshold
    panels = [
        (s_v1_lo, "V1 · LongOnly", "CSI 931018 买入持有（因子列）", "#1f77b4"),
        (s_v2_lo, "V2 · LongOnly", "CSI 931018 买入持有（与 V1 同源）", "#9467bd"),
        (s_v1_ls, "V1 · LongShort", "利差符号切换（CSI）", "#ff7f0e"),
        (s_v2_ls, "V2 · LongShort", f"Dynamic 离散仓位（放宽阈值 ±{th}，默认 0.5）", "#2ca02c"),
    ]

    note = (
        f"<h2>设定</h2><p>V2 SOPConfig.pos_threshold：<strong>{th}</strong>（默认报告为 0.5）。"
        "V1 切换信号与标价不变。</p>"
    )

    write_four_panel_variant_report(
        OUT_DIR,
        idx=idx,
        panels=panels,
        rows_metrics=rows,
        turn_v2=turn_v2,
        report_h1="变体：放宽 V2 出手（四策略分面板）",
        suptitle=f"四策略周频净值｜V2 离散阈值放宽为 ±{th}",
        html_note_p=note,
    )


if __name__ == "__main__":
    main()
