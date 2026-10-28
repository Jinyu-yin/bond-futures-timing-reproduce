"""
合并简报：v1 vs v2 × LongOnly vs LongShort（四条净值曲线 + 指标表），并附日频择时多空胜率。

- v1 LongShort：利差符号切换（public_factors 与 build_public_macro_dataset 一致）
- v2 LongShort：input_data.xlsx 因子筛选 + Dynamic 离散仓位（同 build_ls_gov_cp_report_v2）

输出：reproduce/reports/gov_cp_ls/report_combined/…

说明：单图四线时两条 LongOnly 会重合，看起来像「只有两条线」。若要四条都醒目，请改用
build_ls_gov_cp_report_four_panel.py → report_four_panel/（2×2 分面板）。
"""

from __future__ import annotations

import html
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from build_ls_gov_cp_report_v2 import (
    ann_vol_daily,
    cagr_from_nav,
    compute_v2_daily_strategy_panel,
    max_drawdown,
    sample_years,
)
from gov_cp_timing_metrics import fmt_pct, longonly_buyhold_win_rate, timing_win_rates_gov_cp, timing_win_rates_from_position

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"
OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls" / "report_combined"
OUT_HTML = OUT_DIR / "report.html"
OUT_PDF = OUT_DIR / "report.pdf"
OUT_FIG = OUT_DIR / "fig_four_way.png"
OUT_CSV = OUT_DIR / "summary.csv"

COL_LONG = "5_long_gov_510y_csi931018_close"
COL_SHORT = "5_short_cp_csi_close"
COL_STRAT_LO = "5_strat_longonly_nav"
COL_STRAT_SW = "5_strat_switch_gov_cp_nav"


def _pdf_font() -> str:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def _p(text: str, font: str, st: ParagraphStyle) -> Paragraph:
    safe = html.escape(text or "").replace("\n", "<br/>")
    return Paragraph(safe, st)


def _fit_img(img: Image, max_w: float, max_h: float) -> Image:
    iw = float(img.imageWidth)
    ih = float(img.imageHeight)
    if iw <= 0 or ih <= 0:
        return img
    s = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * s
    img.drawHeight = ih * s
    return img


def v1_weekly_navs(base_daily: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    nav_strat_lo_daily = pd.to_numeric(base_daily[COL_STRAT_LO], errors="coerce")
    lo_wk_ret = pd.DataFrame({"Date": base_daily["Date"], "r": nav_strat_lo_daily.pct_change()}).set_index("Date")
    lo_wk_ret = (1.0 + lo_wk_ret["r"]).resample("W-FRI").prod() - 1.0
    nav_lo = (1.0 + lo_wk_ret.fillna(0.0)).cumprod()

    nav_strat_sw_daily = pd.to_numeric(base_daily[COL_STRAT_SW], errors="coerce")
    sw_wk_ret = pd.DataFrame({"Date": base_daily["Date"], "r": nav_strat_sw_daily.pct_change()}).set_index("Date")
    sw_wk_ret = (1.0 + sw_wk_ret["r"]).resample("W-FRI").prod() - 1.0
    nav_ls = (1.0 + sw_wk_ret.fillna(0.0)).cumprod()
    return nav_lo, nav_ls


def v2_weekly_navs(daily_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    wk = pd.DataFrame(index=daily_df.index)
    wk["LongOnlyRet"] = (1.0 + daily_df["LongOnlyRet"]).resample("W-FRI").prod() - 1.0
    wk["StratRet"] = (1.0 + daily_df["StratRet"]).resample("W-FRI").prod() - 1.0
    wk = wk.dropna(how="all")
    nav_lo = (1.0 + wk["LongOnlyRet"]).cumprod()
    nav_ls = (1.0 + wk["StratRet"]).cumprod()
    return nav_lo, nav_ls


def plot_four_way(df_plot: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(df_plot["Date"], df_plot["v1_lo"], lw=2.0, color="#1f77b4", label="V1 · LongOnly")
    ax.plot(df_plot["Date"], df_plot["v1_ls"], lw=2.0, color="#ff7f0e", linestyle="--", label="V1 · LongShort（利差切换）")
    ax.plot(df_plot["Date"], df_plot["v2_lo"], lw=1.6, color="#9467bd", label="V2 · LongOnly")
    ax.plot(df_plot["Date"], df_plot["v2_ls"], lw=1.6, color="#2ca02c", linestyle=":", label="V2 · LongShort（因子切换）")
    ax.set_title("四策略净值对比（周频，样本交集）｜v1/v2 × LongOnly/LongShort")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值（起点=1）")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.02,
        "V1：利差符号切换；V2：input_data 因子筛选 Dynamic 信号。\n"
        "LongOnly：均为持有 931018（数值在周频对齐下应与 V1/V2 同源）。",
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def main() -> None:
    if not FACTORS_CSV.exists():
        raise FileNotFoundError(FACTORS_CSV)

    base_daily = pd.read_csv(FACTORS_CSV)
    base_daily["Date"] = pd.to_datetime(base_daily["Date"])
    base_daily = base_daily.sort_values("Date")
    for c in (COL_STRAT_LO, COL_STRAT_SW, COL_LONG, COL_SHORT):
        if c not in base_daily.columns:
            raise KeyError(f"因子表缺少 {c}")

    daily_v2, _, _, turn_v2 = compute_v2_daily_strategy_panel()
    nav_lo_v1, nav_ls_v1 = v1_weekly_navs(base_daily)
    nav_lo_v2, nav_ls_v2 = v2_weekly_navs(daily_v2)

    idx = (
        nav_lo_v1.index.intersection(nav_ls_v1.index)
        .intersection(nav_lo_v2.index)
        .intersection(nav_ls_v2.index)
        .sort_values()
    )
    if len(idx) < 5:
        raise RuntimeError("周频样本交集过短，无法生成合并报告")

    y = sample_years(pd.Series(idx))

    def norm_slice(nav: pd.Series) -> pd.Series:
        s = nav.reindex(idx).ffill()
        a0 = float(s.iloc[0])
        return s / a0 if a0 else s

    def pack(s_norm: pd.Series, tag: str) -> dict[str, object]:
        r = s_norm.pct_change().fillna(0.0)
        return {
            "tag": tag,
            "cagr": cagr_from_nav(s_norm, y),
            "vol": ann_vol_daily(r),
            "mdd": max_drawdown(s_norm),
        }

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

    s_v1_lo = norm_slice(nav_lo_v1)
    s_v1_ls = norm_slice(nav_ls_v1)
    s_v2_lo = norm_slice(nav_lo_v2)
    s_v2_ls = norm_slice(nav_ls_v2)

    rows_metrics: list[dict[str, object]] = []
    rows_metrics.append(
        {
            **pack(s_v1_lo, "V1 LongOnly"),
            "wr_long": fmt_pct(tw_lo["timing_long_win_rate"]),
            "wr_short": "—",
        }
    )
    rows_metrics.append(
        {
            **pack(s_v1_ls, "V1 LongShort"),
            "wr_long": fmt_pct(tw_v1_ls["timing_long_win_rate"]),
            "wr_short": fmt_pct(tw_v1_ls["timing_short_win_rate"]),
        }
    )
    rows_metrics.append(
        {
            **pack(s_v2_lo, "V2 LongOnly"),
            "wr_long": fmt_pct(tw_lo["timing_long_win_rate"]),
            "wr_short": "—",
        }
    )
    rows_metrics.append(
        {
            **pack(s_v2_ls, "V2 LongShort"),
            "wr_long": fmt_pct(tw_v2_ls["timing_long_win_rate"]),
            "wr_short": fmt_pct(tw_v2_ls["timing_short_win_rate"]),
        }
    )

    df_plot = pd.DataFrame(
        {
            "Date": idx,
            "v1_lo": s_v1_lo.values,
            "v1_ls": s_v1_ls.values,
            "v2_lo": s_v2_lo.values,
            "v2_ls": s_v2_ls.values,
        }
    )
    plot_four_way(df_plot)

    pd.DataFrame(rows_metrics).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    tbl_rows = ""
    for rm in rows_metrics:
        tbl_rows += (
            f"<tr><td>{html.escape(str(rm['tag']))}</td>"
            f"<td style='text-align:right'>{rm['cagr']*100:.3f}%</td>"
            f"<td style='text-align:right'>{rm['vol']*100:.3f}%</td>"
            f"<td style='text-align:right'>{rm['mdd']*100:.2f}%</td>"
            f"<td style='text-align:right'>{html.escape(str(rm['wr_long']))}</td>"
            f"<td style='text-align:right'>{html.escape(str(rm['wr_short']))}</td></tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>国债/短融：v1 vs v2 四策略对比</title>
<style>
  body {{ font-family: "Microsoft YaHei", SimSun, serif; margin: 0; background: #f5f5f5; }}
  .WordSection1 {{
    max-width: 800px; margin: 24px auto; padding: 48px 56px; background: #fff;
    box-shadow: 0 1px 6px rgba(0,0,0,.08);
  }}
  h1 {{ font-size: 20pt; margin: 0 0 18px 0; color: #0F4761; }}
  h2 {{ font-size: 14pt; margin: 22px 0 10px 0; color: #0F4761; border-bottom: 1px solid #c8d6e5; padding-bottom: 4px; }}
  p {{ font-size: 11pt; line-height: 1.75; text-align: justify; margin: 8px 0; }}
  table.stats {{ border-collapse: collapse; width: 100%; font-size: 10.5pt; margin: 12px 0; }}
  table.stats td, table.stats th {{ border: 1px solid #9fb3c8; padding: 8px 10px; }}
  table.stats th {{ background: #eaf1f8; text-align: left; }}
  .fig {{ margin: 16px 0; text-align: center; }}
  .fig img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  .muted {{ color: #555; font-size: 10pt; }}
  code {{ background: #f0f4f8; padding: 0 4px; }}
</style>
</head>
<body>
<div class="WordSection1">
  <h1>国债活跃券 vs 短融：v1 / v2 × LongOnly / LongShort 合并对比</h1>
  <p class="muted">生成时间：{html.escape(gen)}　｜　公开因子：<code>reproduce/public_factors.csv</code>　｜　v2 信号：<code>input_data.xlsx</code></p>

  <h2>一、说明</h2>
  <p>净值曲线在周频采样<strong>交集区间</strong>内对齐（起点重新归一为 1 便于比较形状）。收益风险指标均在同一交集上计算。</p>
  <p><b>择时胜率（日频）</b>：不考虑票息拆分；持国债日统计国债日收益&gt;0 占比（多头胜率）；持短融日统计短融日收益&gt;0 占比（空头胜率列）。LongOnly 策略空头胜率记为「—」。</p>
  <p>V2 LongShort 换仓次数（日频估计）：<strong>{int((turn_v2 > 1e-12).sum())}</strong>。</p>

  <h2>二、四策略指标与胜率</h2>
  <table class="stats">
    <tr><th>版本 · 策略</th><th>近似年化</th><th>年化波动</th><th>最大回撤</th><th>多头胜率</th><th>空头胜率</th></tr>
    {tbl_rows}
  </table>

  <h2>三、净值对比图</h2>
  <div class="fig"><img src="{OUT_FIG.name}" alt="four-way nav" /></div>

  <h2>四、风险提示</h2>
  <p>简化口径与指数回测，不等于实盘；合并对比仅供方法对照。</p>
</div>
</body>
</html>
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html_doc, encoding="utf-8")

    font = _pdf_font()
    title = ParagraphStyle("t", fontName=font, fontSize=18, leading=24, textColor=colors.HexColor("#0F4761"), spaceAfter=12)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#0F4761"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", fontName=font, fontSize=10.5, leading=17, spaceAfter=6, wordWrap="CJK")
    small = ParagraphStyle("s", fontName=font, fontSize=9, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=8)

    story: list = []
    story.append(_p("国债/短融：v1 vs v2 四策略对比", font, title))
    story.append(_p(f"生成时间：{gen}　数据：public_factors.csv + input_data.xlsx（v2）", font, small))
    story.append(_p("一、四策略指标（周频交集样本）与择时胜率（日频）", font, h2))
    hdr = ["版本·策略", "年化", "波动", "最大回撤", "多头胜率", "空头胜率"]
    data = [[_p(x, font, body) for x in hdr]]
    for rm in rows_metrics:
        data.append(
            [
                _p(str(rm["tag"]), font, body),
                _p(f"{float(rm['cagr'])*100:.3f}%", font, body),
                _p(f"{float(rm['vol'])*100:.3f}%", font, body),
                _p(f"{float(rm['mdd'])*100:.2f}%", font, body),
                _p(str(rm["wr_long"]), font, body),
                _p(str(rm["wr_short"]), font, body),
            ]
        )
    tbl = Table(data, colWidths=[3.2 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb3c8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 10))
    if OUT_FIG.exists():
        story.append(_fit_img(Image(str(OUT_FIG)), max_w=16 * cm, max_h=9.5 * cm))
    story.append(Spacer(1, 8))
    story.append(_p("二、风险提示：仅供研究整理。", font, h2))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build(story)

    themed = ROOT / "111.files" / "themedata.thmx"
    if themed.exists():
        shutil.copy2(themed, OUT_DIR / "themedata_ref.thmx")

    print("saved", OUT_HTML)
    print("saved", OUT_PDF)
    print("saved", OUT_FIG)
    print("saved", OUT_CSV)


if __name__ == "__main__":
    main()
