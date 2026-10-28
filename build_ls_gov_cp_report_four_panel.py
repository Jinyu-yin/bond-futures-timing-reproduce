"""
四策略合并简报（真正「四条」可见）：2×2 子图，每格一条净值曲线。

背景：单轴叠加时 V1/V2 的 LongOnly 完全同源会重合，易被误认为只有 2 条线。
本脚本输出到新目录 report_four_panel，不覆盖 report_combined。

运行：py reproduce/build_ls_gov_cp_report_four_panel.py
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
OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls" / "report_four_panel"
OUT_HTML = OUT_DIR / "report.html"
OUT_PDF = OUT_DIR / "report.pdf"
OUT_FIG = OUT_DIR / "fig_four_panel_2x2.png"
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


def plot_four_panel_2x2(dates: pd.DatetimeIndex, s_v1_lo: pd.Series, s_v2_lo: pd.Series, s_v1_ls: pd.Series, s_v2_ls: pd.Series) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    panels = [
        (axes[0, 0], s_v1_lo, "V1 · LongOnly", "买入持有 931018（public 因子列）", "#1f77b4"),
        (axes[0, 1], s_v2_lo, "V2 · LongOnly", "买入持有 931018（与 V1 同源标价）", "#9467bd"),
        (axes[1, 0], s_v1_ls, "V1 · LongShort", "利差 rL−rS 符号切换", "#ff7f0e"),
        (axes[1, 1], s_v2_ls, "V2 · LongShort", "input_data 因子 Dynamic 切换", "#2ca02c"),
    ]
    for ax, series, title, subtitle, color in panels:
        ax.plot(dates, series.values, lw=2.0, color=color)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="#bbbbbb", lw=0.8, linestyle="--")
        ax.set_ylabel("净值（交集起点=1）")
    axes[1, 0].set_xlabel("日期")
    axes[1, 1].set_xlabel("日期")
    fig.suptitle("四策略净值（周频交集）｜分面板展示，避免 LongOnly 重合误判为单线", fontsize=12, y=1.02)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
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
        raise RuntimeError("周频样本交集过短，无法生成报告")

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

    plot_four_panel_2x2(idx, s_v1_lo, s_v2_lo, s_v1_ls, s_v2_ls)

    rows_metrics: list[dict[str, object]] = [
        {**pack(s_v1_lo, "V1 LongOnly"), "wr_long": fmt_pct(tw_lo["timing_long_win_rate"]), "wr_short": "—"},
        {**pack(s_v1_ls, "V1 LongShort"), "wr_long": fmt_pct(tw_v1_ls["timing_long_win_rate"]), "wr_short": fmt_pct(tw_v1_ls["timing_short_win_rate"])},
        {**pack(s_v2_lo, "V2 LongOnly"), "wr_long": fmt_pct(tw_lo["timing_long_win_rate"]), "wr_short": "—"},
        {**pack(s_v2_ls, "V2 LongShort"), "wr_long": fmt_pct(tw_v2_ls["timing_long_win_rate"]), "wr_short": fmt_pct(tw_v2_ls["timing_short_win_rate"])},
    ]

    pd.DataFrame(rows_metrics).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    tbl_rows = ""
    for rm in rows_metrics:
        tbl_rows += (
            f"<tr><td>{html.escape(str(rm['tag']))}</td>"
            f"<td style='text-align:right'>{float(rm['cagr'])*100:.3f}%</td>"
            f"<td style='text-align:right'>{float(rm['vol'])*100:.3f}%</td>"
            f"<td style='text-align:right'>{float(rm['mdd'])*100:.2f}%</td>"
            f"<td style='text-align:right'>{html.escape(str(rm['wr_long']))}</td>"
            f"<td style='text-align:right'>{html.escape(str(rm['wr_short']))}</td></tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>四策略分面板对比（v1/v2×LO/LS）</title>
<style>
  body {{ font-family: "Microsoft YaHei", SimSun, serif; margin: 0; background: #f5f5f5; }}
  .WordSection1 {{
    max-width: 900px; margin: 24px auto; padding: 48px 56px; background: #fff;
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
  <h1>四策略对比（分面板 2×2）</h1>
  <p class="muted">生成时间：{html.escape(gen)}　｜　输出目录：<code>reproduce/reports/gov_cp_ls/report_four_panel/</code></p>

  <h2>一、为何单独做本页</h2>
  <p>单图四线叠加时，<b>V1 LongOnly</b> 与 <b>V2 LongOnly</b> 均为持有同一标的（931018），曲线会<strong>完全重合</strong>，
  肉眼往往只看到「两条 LongShort」或误以为只有两条线。这里用 <strong>四个子图</strong>，每格一条曲线，对应关系一目了然。</p>
  <p>V2 LongShort 换仓次数（日频估计）：<strong>{int((turn_v2 > 1e-12).sum())}</strong>。</p>

  <h2>二、指标与择时胜率（周频交集 / 日频胜率）</h2>
  <table class="stats">
    <tr><th>版本 · 策略</th><th>近似年化</th><th>年化波动</th><th>最大回撤</th><th>多头胜率</th><th>空头胜率</th></tr>
    {tbl_rows}
  </table>

  <h2>三、2×2 净值图（每子图一条线）</h2>
  <div class="fig"><img src="{OUT_FIG.name}" alt="four panel nav" /></div>

  <h2>四、风险提示</h2>
  <p>简化回测口径，仅供研究对照。</p>
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
    story.append(_p("四策略对比（分面板 2×2）", font, title))
    story.append(_p(f"生成时间：{gen}　单图叠加时 LongOnly 重合，本分面板展示四条。", font, small))
    story.append(_p("一、指标表", font, h2))
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
    story.append(_p("二、2×2 净值图（见插图）", font, h2))
    if OUT_FIG.exists():
        story.append(_fit_img(Image(str(OUT_FIG)), max_w=17 * cm, max_h=12 * cm))
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
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
