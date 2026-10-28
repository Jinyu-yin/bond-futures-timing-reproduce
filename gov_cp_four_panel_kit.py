"""四面板 gov/cp 报告共用：周频净值、作图、HTML/PDF。"""

from __future__ import annotations

import html
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from build_ls_gov_cp_report_v2 import ann_vol_daily, cagr_from_nav, max_drawdown, sample_years
from gov_cp_continuous_weights import bipartite_weights_from_z, daily_strategy_return_continuous
from gov_cp_timing_metrics import fmt_pct, longonly_buyhold_win_rate

COL_LONG = "5_long_gov_510y_csi931018_close"
COL_SHORT = "5_short_cp_csi_close"
COL_STRAT_LO = "5_strat_longonly_nav"
COL_STRAT_SW = "5_strat_switch_gov_cp_nav"


def v1_weekly_navs_from_factors_csv(base_daily: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    nav_strat_lo_daily = pd.to_numeric(base_daily[COL_STRAT_LO], errors="coerce")
    lo_wk_ret = pd.DataFrame({"Date": base_daily["Date"], "r": nav_strat_lo_daily.pct_change()}).set_index("Date")
    lo_wk_ret = (1.0 + lo_wk_ret["r"]).resample("W-FRI").prod() - 1.0
    nav_lo = (1.0 + lo_wk_ret.fillna(0.0)).cumprod()

    nav_strat_sw_daily = pd.to_numeric(base_daily[COL_STRAT_SW], errors="coerce")
    sw_wk_ret = pd.DataFrame({"Date": base_daily["Date"], "r": nav_strat_sw_daily.pct_change()}).set_index("Date")
    sw_wk_ret = (1.0 + sw_wk_ret["r"]).resample("W-FRI").prod() - 1.0
    nav_ls = (1.0 + sw_wk_ret.fillna(0.0)).cumprod()
    return nav_lo, nav_ls


def v2_weekly_navs_from_daily(daily_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    wk = pd.DataFrame(index=daily_df.index)
    wk["LongOnlyRet"] = (1.0 + daily_df["LongOnlyRet"]).resample("W-FRI").prod() - 1.0
    wk["StratRet"] = (1.0 + daily_df["StratRet"]).resample("W-FRI").prod() - 1.0
    wk = wk.dropna(how="all")
    nav_lo = (1.0 + wk["LongOnlyRet"]).cumprod()
    nav_ls = (1.0 + wk["StratRet"]).cumprod()
    return nav_lo, nav_ls


def v1_switch_daily_leveraged_panel(
    base_daily: pd.DataFrame,
    r_gov: pd.Series,
    r_cp: pd.Series,
    long_weight: float,
    short_weight: float,
    slip: float = 0.0001,
) -> pd.DataFrame:
    """与 build_public_macro_dataset 切换逻辑一致，收益换为 r_gov/r_cp，并乘杠杆。"""
    bd = base_daily.sort_values("Date").copy()
    idx = pd.DatetimeIndex(pd.to_datetime(bd["Date"]))
    wg = pd.to_numeric(bd["5_pos_switch_gov_weight"], errors="coerce").fillna(0.0).values
    wc = pd.to_numeric(bd["5_pos_switch_cp_weight"], errors="coerce").fillna(0.0).values
    w_gov = pd.Series(wg, index=idx)
    w_cp = pd.Series(wc, index=idx)

    rg = pd.to_numeric(r_gov.reindex(idx), errors="coerce").ffill().bfill().fillna(0.0)
    rs = pd.to_numeric(r_cp.reindex(idx), errors="coerce").ffill().bfill().fillna(0.0)

    wg_e = w_gov.shift(1).fillna(0.0)
    wc_e = w_cp.shift(1).fillna(0.0)
    lw = float(long_weight)
    sw = float(short_weight)
    strat = wg_e * lw * rg + wc_e * sw * rs
    traded = ((wg_e.diff().abs() + wc_e.diff().abs()) > 1e-12).astype(float).fillna(0.0)
    strat = strat - slip * traded
    lo = lw * rg
    pos = pd.Series(0.0, index=idx)
    pos[w_gov > 0.5] = 1.0
    pos[w_cp > 0.5] = -1.0
    return pd.DataFrame({"r_gov": rg, "r_cp": rs, "LongOnlyRet": lo, "StratRet": strat, "Pos": pos})


def v1_continuous_daily_panel(
    base_daily: pd.DataFrame,
    r_gov: pd.Series,
    r_cp: pd.Series,
    *,
    cont_gamma: float = 1.0,
    w_floor: float = 0.8,
    w_cap: float = 1.2,
    diff_mean_win: int = 20,
    diff_sd_win: int = 60,
    slip: float = 0.0001,
    long_weight: float = 1.0,
    short_weight: float = 1.0,
) -> pd.DataFrame:
    """
    V1 连续仓位：用利差日收益 (r_gov−r_cp) 的滚动 Z 作信号，映射为 w_gov,w_cp∈[w_floor,w_cap]（和为 w_floor+w_cap），
    再乘 long_weight/short_weight 缩放两侧名义（默认 1）。
    """
    bd = base_daily.sort_values("Date").copy().drop_duplicates(subset=["Date"], keep="last")
    idx = pd.DatetimeIndex(pd.to_datetime(bd["Date"]))
    rg = pd.to_numeric(r_gov.reindex(idx), errors="coerce").ffill().bfill().fillna(0.0)
    rs = pd.to_numeric(r_cp.reindex(idx), errors="coerce").ffill().bfill().fillna(0.0)
    diff = rg - rs
    z = diff.rolling(diff_mean_win, min_periods=max(5, diff_mean_win // 4)).mean() / (
        diff.rolling(diff_sd_win, min_periods=max(10, diff_sd_win // 4)).std() + 1e-8
    )
    z = z.clip(-6, 6).fillna(0.0)
    w_gov, w_cp = bipartite_weights_from_z(z, gamma=cont_gamma, w_floor=w_floor, w_cap=w_cap)
    w_gov = w_gov * float(long_weight)
    w_cp = w_cp * float(short_weight)
    strat_ret, wg_e, wc_e = daily_strategy_return_continuous(rg, rs, w_gov, w_cp, slip_per_unit_turnover=slip)
    lo = float(long_weight) * rg
    net = w_gov - w_cp
    pos = pd.Series(np.sign(net).replace(0.0, np.nan), index=idx).fillna(0.0)
    return pd.DataFrame(
        {
            "r_gov": rg,
            "r_cp": rs,
            "LongOnlyRet": lo,
            "StratRet": strat_ret,
            "Pos": pos,
            "w_gov_exe": wg_e,
            "w_cp_exe": wc_e,
        }
    )


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


def plot_four_panel(
    out_fig: Path,
    dates: pd.DatetimeIndex,
    *,
    suptitle: str,
    panels: list[tuple[pd.Series, str, str, str]],
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    ax_flat = axes.flatten()
    for ax, (series, title, subtitle, color) in zip(ax_flat, panels):
        ax.plot(dates, series.values, lw=2.0, color=color)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color="#bbbbbb", lw=0.8, linestyle="--")
        ax.set_ylabel("净值（交集起点=1）")
    axes[1, 0].set_xlabel("日期")
    axes[1, 1].set_xlabel("日期")
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_four_panel_variant_report(
    out_dir: Path,
    *,
    idx: pd.DatetimeIndex,
    panels: list[tuple[pd.Series, str, str, str]],
    rows_metrics: list[dict[str, object]],
    turn_v2: pd.Series,
    report_h1: str,
    suptitle: str,
    html_note_p: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_html = out_dir / "report.html"
    out_pdf = out_dir / "report.pdf"
    out_fig = out_dir / "fig_four_panel_2x2.png"
    out_csv = out_dir / "summary.csv"

    plot_four_panel(out_fig, idx, suptitle=suptitle, panels=panels)

    pd.DataFrame(rows_metrics).to_csv(out_csv, index=False, encoding="utf-8-sig")

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
<title>{html.escape(report_h1)}</title>
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
  <h1>{html.escape(report_h1)}</h1>
  <p class="muted">生成时间：{html.escape(gen)}　｜　文件夹：<code>reproduce/reports/gov_cp_ls/{html.escape(out_dir.name)}</code></p>
  {html_note_p}
  <p>V2 LongShort 换仓次数（日频估计）：<strong>{int((turn_v2 > 1e-12).sum())}</strong>。</p>

  <h2>指标与择时胜率（周频交集 / 日频胜率）</h2>
  <table class="stats">
    <tr><th>版本 · 策略</th><th>近似年化</th><th>年化波动</th><th>最大回撤</th><th>多头胜率</th><th>空头胜率</th></tr>
    {tbl_rows}
  </table>

  <h2>2×2 净值图（每子图一条线）</h2>
  <div class="fig"><img src="{out_fig.name}" alt="four panel" /></div>

  <h2>风险提示</h2>
  <p>简化回测口径，仅供研究对照。</p>
</div>
</body>
</html>
"""
    out_html.write_text(html_doc, encoding="utf-8")

    font = _pdf_font()
    title = ParagraphStyle("t", fontName=font, fontSize=18, leading=24, textColor=colors.HexColor("#0F4761"), spaceAfter=12)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#0F4761"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", fontName=font, fontSize=10.5, leading=17, spaceAfter=6, wordWrap="CJK")
    small = ParagraphStyle("s", fontName=font, fontSize=9, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=8)

    story: list = []
    story.append(_p(report_h1, font, title))
    story.append(_p(f"生成时间：{gen}", font, small))
    story.append(_p("指标表（周频交集）与择时胜率（日频）", font, h2))
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
    story.append(_p("2×2 净值图", font, h2))
    if out_fig.exists():
        story.append(_fit_img(Image(str(out_fig)), max_w=17 * cm, max_h=12 * cm))

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )
    doc.build(story)

    project_root = out_dir.parents[3]
    themed = project_root / "111.files" / "themedata.thmx"
    if themed.exists():
        shutil.copy2(themed, out_dir / "themedata_ref.thmx")

    print("saved", out_html)
    print("saved", out_pdf)
    print("saved", out_fig)
    print("saved", out_csv)


def build_metrics_rows(
    *,
    idx: pd.DatetimeIndex,
    s_v1_lo: pd.Series,
    s_v1_ls: pd.Series,
    s_v2_lo: pd.Series,
    s_v2_ls: pd.Series,
    tw_lo: dict,
    tw_v1_ls: dict,
    tw_v2_ls: dict,
) -> list[dict[str, object]]:
    y = sample_years(pd.Series(idx))

    def pack(s_norm: pd.Series, tag: str) -> dict[str, object]:
        r = s_norm.pct_change().fillna(0.0)
        return {"tag": tag, "cagr": cagr_from_nav(s_norm, y), "vol": ann_vol_daily(r), "mdd": max_drawdown(s_norm)}

    return [
        {
            **pack(s_v1_lo, "V1 LongOnly"),
            "wr_long": fmt_pct(tw_lo["timing_long_win_rate"]),
            "wr_short": "—",
        },
        {
            **pack(s_v1_ls, "V1 LongShort"),
            "wr_long": fmt_pct(tw_v1_ls["timing_long_win_rate"]),
            "wr_short": fmt_pct(tw_v1_ls["timing_short_win_rate"]),
        },
        {
            **pack(s_v2_lo, "V2 LongOnly"),
            "wr_long": fmt_pct(tw_lo["timing_long_win_rate"]),
            "wr_short": "—",
        },
        {
            **pack(s_v2_ls, "V2 LongShort"),
            "wr_long": fmt_pct(tw_v2_ls["timing_long_win_rate"]),
            "wr_short": fmt_pct(tw_v2_ls["timing_short_win_rate"]),
        },
    ]
