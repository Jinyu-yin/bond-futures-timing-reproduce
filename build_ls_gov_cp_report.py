"""
生成「国债活跃券 vs 中证短融」多空利差简报（HTML + PDF），含 **LongOnly** 与 **LongShort** 两条最终策略表与对比图。

版式：段落标题层级与插图排布参考项目根目录下 Word 导出的 `111.htm` 及其资源目录
`111.files/`（HTML 为 UTF-8 独立网页；PDF 由 ReportLab 生成，中文为内置 STSong-Light）。

依赖：先运行 `py reproduce/build_public_macro_dataset.py` 更新 `public_factors.csv`。
"""

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

from gov_cp_timing_metrics import fmt_pct, longonly_buyhold_win_rate, timing_win_rates_gov_cp

ROOT = Path(__file__).resolve().parents[1]
FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"
OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls"
OUT_HTML = OUT_DIR / "report.html"
OUT_PDF = OUT_DIR / "report.pdf"
OUT_FIG = OUT_DIR / "fig_cumnav.png"
OUT_FIG_STRAT = OUT_DIR / "fig_strategies_longonly_longshort.png"
OUT_STRAT_CSV = OUT_DIR / "strategies_nav.csv"

COL_LONG = "5_long_gov_510y_csi931018_close"
COL_SHORT = "5_short_cp_csi_close"
COL_LS = "5_ls_gov510y_minus_shortcp_nav"
COL_HEDGE = "5_hedge_rollbeta60_ls_nav"
COL_STRAT_LO = "5_strat_longonly_nav"
COL_STRAT_LS = "5_strat_longshort_nav"
COL_STRAT_SW = "5_strat_switch_gov_cp_nav"


def _fit_img(img: Image, max_w: float, max_h: float) -> Image:
    """Keep aspect ratio, fit into (max_w, max_h)."""
    iw = float(img.imageWidth)
    ih = float(img.imageHeight)
    if iw <= 0 or ih <= 0:
        return img
    s = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * s
    img.drawHeight = ih * s
    return img


def level_to_cumnav(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if not s.notna().any():
        return s
    first = float(s.dropna().iloc[0])
    if first == 0:
        return s * float("nan")
    return s / first


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


def max_drawdown(nav: pd.Series) -> float:
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if len(nav) < 2:
        return float("nan")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def plot_figure(
    df: pd.DataFrame,
    nav_long: pd.Series,
    nav_short: pd.Series,
    nav_ls: pd.Series,
    metrics: dict,
    nav_hedge: pd.Series | None,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(df["Date"], nav_long, lw=1.2, color="#1f77b4", label=f"国债5-10y活跃(931018)  CAGR≈{metrics['c_long']*100:.2f}%")
    ax.plot(df["Date"], nav_short, lw=1.2, color="#ff7f0e", label=f"中证短融(H11014)  CAGR≈{metrics['c_short']*100:.2f}%")
    ax.plot(df["Date"], nav_ls, lw=1.4, color="black", alpha=0.85, label=f"1:1 多空利差  CAGR≈{metrics['c_ls']*100:.2f}%")
    if nav_hedge is not None and nav_hedge.notna().any():
        ch = metrics.get("c_hedge", float("nan"))
        cstr = f"{ch * 100:.2f}%" if ch == ch and np.isfinite(ch) else "N/A"
        ax.plot(
            df["Date"],
            nav_hedge,
            lw=1.35,
            color="#2ca02c",
            linestyle="--",
            label=f"滚动β对冲(60日)  CAGR≈{cstr}",
        )
    ax.set_title("累计净值（起点=1）｜国债 vs 短融：等额多空与滚动β对冲")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    note = (
        f"样本跨度≈{metrics['years']:.2f}年。黑线：等额 1:1 多空（rL−rS）；"
        "绿虚线：滚动 60 日估计 β，用前一日 β 做 rL−β·rS（降低短融方向暴露）。"
        "均未计交易费用与保证金。"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35))
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def plot_strategies_figure(
    df: pd.DataFrame,
    nav_lo: pd.Series,
    nav_sw: pd.Series | None,
    metrics: dict,
) -> None:
    """最终策略：纯多 vs 切换策略，单独成图便于写入报告。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 5.2))
    clo = metrics.get("strat_lo_cagr", float("nan"))
    csw = metrics.get("strat_sw_cagr", float("nan"))
    l1 = f"LongOnly（931018） CAGR≈{clo * 100:.2f}%" if clo == clo and np.isfinite(clo) else "LongOnly（931018）"
    l3 = f"LongShort（Gov/CP切换） CAGR≈{csw * 100:.2f}%" if csw == csw and np.isfinite(csw) else "LongShort（Gov/CP切换）"
    ax.plot(df["Date"], nav_lo, lw=2.2, color="#1f77b4", label=l1)
    if nav_sw is not None and nav_sw.notna().any():
        ax.plot(df["Date"], nav_sw, lw=2.2, color="#2ca02c", label=l3)
    ax.set_title("策略净值对比（起点=1）｜LongOnly vs LongShort")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.02,
        "LongOnly：仅持有国债活跃券指数；"
        "LongShort：按信号在国债/短融之间二选一切换（含固定滑点）。",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="#f5f5dc", alpha=0.45),
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_STRAT, dpi=150)
    plt.close(fig)


def build_html(df: pd.DataFrame, metrics: dict) -> str:
    rows = [
        ("样本跨度（近似年）", f"{metrics['years']:.3f}"),
        ("国债多头（931018）近似年化", f"{metrics['c_long']*100:.3f}%"),
        ("短融（H11014）近似年化", f"{metrics['c_short']*100:.3f}%"),
        ("1:1 多空利差净值近似年化", f"{metrics['c_ls']*100:.3f}%"),
        ("1:1 利差日收益年化波动", f"{metrics['vol_ls']*100:.3f}%"),
        ("国债归一净值最大回撤", f"{metrics['mdd_long']*100:.2f}%"),
        ("短融归一净值最大回撤", f"{metrics['mdd_short']*100:.2f}%"),
        ("1:1 利差净值最大回撤", f"{metrics['mdd_ls']*100:.2f}%"),
        ("国债与短融日收益相关系数", f"{metrics['corr']:.4f}"),
    ]
    if "c_hedge" in metrics and metrics["c_hedge"] == metrics["c_hedge"] and np.isfinite(metrics["c_hedge"]):
        rows.extend(
            [
                ("滚动β对冲（60日）净值近似年化", f"{metrics['c_hedge']*100:.3f}%"),
                ("β对冲日收益年化波动", f"{metrics['vol_hedge']*100:.3f}%"),
                ("β对冲净值最大回撤", f"{metrics['mdd_hedge']*100:.2f}%"),
            ]
        )
    tr = "".join(
        f"<tr><td>{html.escape(a)}</td><td style='text-align:right'>{html.escape(b)}</td></tr>"
        for a, b in rows
    )
    strat_html = ""
    if "strat_lo_cagr" in metrics:
        slo = metrics["strat_lo_cagr"] * 100
        vlo = metrics["strat_lo_vol"] * 100
        mdlo = metrics["strat_lo_mdd"] * 100
        wr_lo_l = metrics.get("strat_wr_lo_long", "—")
        wr_lo_s = metrics.get("strat_wr_lo_short", "—")
        sw_row = ""
        sw_note = ""
        if "strat_sw_cagr" in metrics and metrics["strat_sw_cagr"] == metrics["strat_sw_cagr"] and np.isfinite(metrics["strat_sw_cagr"]):
            ssw = metrics["strat_sw_cagr"] * 100
            vsw = metrics["strat_sw_vol"] * 100
            mdsw = metrics["strat_sw_mdd"] * 100
            slips = metrics.get("strat_sw_slip", 0.0001)
            trades = metrics.get("strat_sw_trades", float("nan"))
            wr_ls_l = metrics.get("strat_wr_ls_long", "—")
            wr_ls_s = metrics.get("strat_wr_ls_short", "—")
            sw_row = (
                f'<tr><td>LongShort</td><td>多国债/多短融二选一切换，含滑点 {slips*100:.2f}%/次</td>'
                f'<td style="text-align:right">{ssw:.3f}%</td>'
                f'<td style="text-align:right">{vsw:.3f}%</td>'
                f'<td style="text-align:right">{mdsw:.2f}%</td>'
                f'<td style="text-align:right">{wr_ls_l}</td>'
                f'<td style="text-align:right">{wr_ls_s}</td></tr>'
            )
            if trades == trades and np.isfinite(trades):
                sw_note = f"<p class='muted'>LongShort（切换）样本期换仓次数≈{int(trades)}（按日频权重变化计 1 次）。</p>"
        strat_html = f"""
  <h2>三、最终策略：LongOnly 与 LongShort（v1）</h2>
  <p>策略净值序列已写入因子表列 <code>5_strat_longonly_nav</code>（纯多）与 <code>5_strat_switch_gov_cp_nav</code>（切换策略，含滑点）。同目录另存 <code>strategies_nav.csv</code> 便于复核。</p>
  <p class="muted">择时胜率（日频、不含票息拆分）：持国债日计国债日收益&gt;0 占比；持短融日计短融日收益&gt;0 占比。LongOnly 仅填多头胜率列。</p>
  <table class="stats">
    <tr><th>策略</th><th>口径</th><th>近似年化</th><th>年化波动</th><th>最大回撤</th><th>多头胜率</th><th>空头胜率</th></tr>
    <tr><td>LongOnly</td><td>买入并持有中证 931018，归一到 1</td><td style="text-align:right">{slo:.3f}%</td><td style="text-align:right">{vlo:.3f}%</td><td style="text-align:right">{mdlo:.2f}%</td><td style="text-align:right">{wr_lo_l}</td><td style="text-align:right">{wr_lo_s}</td></tr>
    {sw_row}
  </table>
  <div class="fig"><img src="{OUT_FIG_STRAT.name}" alt="Strategies comparison" /></div>
  {sw_note}
"""
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    ref111 = html.escape("版式与插图层次参考项目根目录下由 Word 导出的 111.htm（配套资源见 111.files/）。")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>国债活跃券与短融多空利差简报</title>
<style>
  body {{ font-family: "Microsoft YaHei", SimSun, serif; margin: 0; background: #f5f5f5; }}
  .WordSection1 {{
    max-width: 720px; margin: 24px auto; padding: 48px 56px; background: #fff;
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
  <h1>国债活跃券与短融：多空利差简报</h1>
  <p class="muted">生成时间：{html.escape(gen)}　｜　数据文件：<code>reproduce/public_factors.csv</code></p>

  <h2>摘要</h2>
  <p>本报告给出中证口径下「中长期国债（活跃券）」与「中证短融」相关分析，并输出两条<strong>最终策略</strong>净值：
  <b>LongOnly</b>（仅持有 931018）与 <b>LongShort</b>（按信号在国债/短融间二选一切换，含滑点）。长端<strong>不使用中债指数</strong>；
  国债为<strong>5–10 年期国债活跃券（931018）</strong>；短融为<strong>中证短融（H11014）</strong>。</p>

  <h2>一、构建口径与数据来源</h2>
  <p><b>（1）做多参考（国债）</b>：中证指数公司发布的「中证5–10年期国债活跃券指数」，代码 <code>931018</code>，
  取日频<strong>收盘</strong>点位，经 akshare <code>stock_zh_index_hist_csindex</code> 拉取后对期货交易日历前向填充对齐。</p>
  <p><b>（2）做空/对冲参考（短融）</b>：中证短融指数 <code>H11014</code> 收盘点位；指数本身为多头口径，在多空分析中常作为短端或对冲标的。</p>
  <p><b>（3）1:1 多空利差（简化对冲）</b>：日收益 rLS = rL − rS，再累乘 (1+rLS)，等价于<strong>等额名义</strong>下「多国债指数、空短融指数」、
  <strong>日末再平衡</strong>、无保证金与费用的理想化口径；属于最常见的「多空对冲」示意之一，但不是最小方差或 β 中性等更复杂策略。</p>
  <p><b>（4）滚动 β 多空对冲（示意）</b>：在 60 个交易日窗口内估计 Cov(rL,rS)/Var(rS)，用<strong>前一日</strong> β 构造
  rH = rL − β·rS，再累乘得到第二条对冲净值（因子表列 <code>5_hedge_rollbeta60_ls_nav</code>）。用于部分对冲短融方向暴露，仍不含交易成本。</p>

  <h2>二、样本区间与收益风险指标</h2>
  <p>样本起止：<strong>{html.escape(str(df['Date'].iloc[0].date()))}</strong> 至
  <strong>{html.escape(str(df['Date'].iloc[-1].date()))}</strong>。</p>
  <table class="stats">
    <tr><th>指标</th><th>数值</th></tr>
    {tr}
  </table>
{strat_html}
  <h2>四、附图：指数与利差全景</h2>
  <div class="fig"><img src="{OUT_FIG.name}" alt="累计净值" /></div>

  <h2>五、版式参考说明</h2>
  <p>{ref111}</p>
  <p>本页为独立 UTF-8 HTML；若需在 Word 中进一步排版，可将本报告另存或嵌入。</p>

  <h2>六、风险提示</h2>
  <p>指数编制方法、样本券更替与再投资假设与真实可交易组合存在差异；与国债期货（如 T）的久期、杠杆与交割机制亦不一致。
  本报告仅供研究整理，不构成投资建议。</p>
</div>
</body>
</html>
"""


def _pdf_font() -> str:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def _p(text: str, font: str, st: ParagraphStyle) -> Paragraph:
    safe = html.escape(text or "").replace("\n", "<br/>")
    return Paragraph(safe, st)


def build_pdf(df: pd.DataFrame, metrics: dict) -> None:
    font = _pdf_font()
    title = ParagraphStyle("t", fontName=font, fontSize=18, leading=24, textColor=colors.HexColor("#0F4761"), spaceAfter=12)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#0F4761"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", fontName=font, fontSize=10.5, leading=17, spaceAfter=6, wordWrap="CJK")
    small = ParagraphStyle("s", fontName=font, fontSize=9, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=8)

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    story: list = []
    story.append(_p("国债活跃券与短融：多空利差简报", font, title))
    story.append(_p(f"生成时间：{gen}　数据：reproduce/public_factors.csv", font, small))

    story.append(_p("摘要", font, h2))
    story.append(
        _p(
            "本报告给出中证口径下中长期国债（活跃券）与中证短融两条指数归一净值及日收益差累计的利差净值。"
            "长端不使用中债指数，采用中证5–10年期国债活跃券（931018）；短端为中证短融（H11014）。",
            font,
            body,
        )
    )

    story.append(_p("一、构建口径与数据来源", font, h2))
    story.append(
        _p(
            "（1）做多参考（国债）：中证5–10年期国债活跃券指数 931018 收盘点位，"
            "经 akshare stock_zh_index_hist_csindex 获取，与期货交易日历对齐后前向填充。",
            font,
            body,
        )
    )
    story.append(
        _p(
            "（2）短融参考：中证短融指数 H11014 收盘；指数为多头口径，多空分析中常作短端或对冲参考。",
            font,
            body,
        )
    )
    story.append(
        _p(
            "（3）1:1 多空：日收益 r = 国债日收益 − 短融日收益，再累乘；等额名义、日末再平衡、无费用与保证金。",
            font,
            body,
        )
    )
    story.append(
        _p(
            "（4）滚动 β 对冲：60 日窗口估计 β，用前一日 β 得 r = rL−β·rS 再累乘；列名 5_hedge_rollbeta60_ls_nav。",
            font,
            body,
        )
    )

    story.append(_p("二、样本区间与指标", font, h2))
    story.append(
        _p(
            f"样本起止：{df['Date'].iloc[0].date()} 至 {df['Date'].iloc[-1].date()}。",
            font,
            body,
        )
    )
    rows = [
        ["样本跨度（近似年）", f"{metrics['years']:.3f}"],
        ["国债多头（931018）近似年化", f"{metrics['c_long']*100:.3f}%"],
        ["短融（H11014）近似年化", f"{metrics['c_short']*100:.3f}%"],
        ["1:1 多空利差净值近似年化", f"{metrics['c_ls']*100:.3f}%"],
        ["1:1 利差日收益年化波动", f"{metrics['vol_ls']*100:.3f}%"],
        ["国债归一净值最大回撤", f"{metrics['mdd_long']*100:.2f}%"],
        ["短融归一净值最大回撤", f"{metrics['mdd_short']*100:.2f}%"],
        ["1:1 利差净值最大回撤", f"{metrics['mdd_ls']*100:.2f}%"],
        ["国债与短融日收益相关系数", f"{metrics['corr']:.4f}"],
    ]
    if "c_hedge" in metrics and metrics["c_hedge"] == metrics["c_hedge"] and np.isfinite(metrics["c_hedge"]):
        rows.extend(
            [
                ["滚动β对冲（60日）净值近似年化", f"{metrics['c_hedge']*100:.3f}%"],
                ["β对冲日收益年化波动", f"{metrics['vol_hedge']*100:.3f}%"],
                ["β对冲净值最大回撤", f"{metrics['mdd_hedge']*100:.2f}%"],
            ]
        )
    tbl_data = [[_p("指标", font, body), _p("数值", font, body)]] + [
        [_p(a, font, body), _p(b, font, body)] for a, b in rows
    ]
    tbl = Table(tbl_data, colWidths=[8.5 * cm, 8.5 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb3c8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 8))

    if "strat_lo_cagr" in metrics:
        story.append(_p("三、最终策略 LongOnly 与 LongShort", font, h2))
        story.append(
            _p(
                "LongOnly：因子列 5_strat_longonly_nav；LongShort：5_strat_switch_gov_cp_nav（按信号在国债/短融之间切换，含固定滑点）。",
                font,
                body,
            )
        )
        st_hdr = ["策略", "口径", "近似年化", "年化波动", "最大回撤", "多头胜率", "空头胜率"]
        st_r1 = [
            "LongOnly",
            "买入并持有931018归一",
            f"{metrics['strat_lo_cagr']*100:.3f}%",
            f"{metrics['strat_lo_vol']*100:.3f}%",
            f"{metrics['strat_lo_mdd']*100:.2f}%",
            metrics.get("strat_wr_lo_long", "—"),
            metrics.get("strat_wr_lo_short", "—"),
        ]
        st_rows = [st_r1]
        if "strat_sw_cagr" in metrics and metrics["strat_sw_cagr"] == metrics["strat_sw_cagr"] and np.isfinite(metrics["strat_sw_cagr"]):
            slip = metrics.get("strat_sw_slip", 0.0001)
            st_rows.append(
                [
                    "LongShort",
                    f"多国债/多短融二选一切换（滑点{slip*100:.2f}%/次）",
                    f"{metrics['strat_sw_cagr']*100:.3f}%",
                    f"{metrics['strat_sw_vol']*100:.3f}%",
                    f"{metrics['strat_sw_mdd']*100:.2f}%",
                    metrics.get("strat_wr_ls_long", "—"),
                    metrics.get("strat_wr_ls_short", "—"),
                ]
            )
        st_data = [[_p(x, font, body) for x in st_hdr]] + [
            [_p(x, font, body) for x in row] for row in st_rows
        ]
        st_tbl = Table(st_data, colWidths=[2.0 * cm, 4.8 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.9 * cm, 1.9 * cm])
        st_tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb3c8")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(st_tbl)
        story.append(Spacer(1, 6))
        if OUT_FIG_STRAT.exists():
            img_s = _fit_img(Image(str(OUT_FIG_STRAT)), max_w=16 * cm, max_h=9.5 * cm)
            story.append(img_s)
        story.append(Spacer(1, 8))

    story.append(_p("四、附图：指数与利差全景", font, h2))
    if OUT_FIG.exists():
        img = _fit_img(Image(str(OUT_FIG)), max_w=16 * cm, max_h=11.0 * cm)
        story.append(img)

    story.append(Spacer(1, 6))
    story.append(_p("五、版式说明", font, h2))
    story.append(_p("段落层级与插图排布参考项目根目录 Word 导出之 111.htm（资源见 111.files）。", font, body))
    story.append(_p("六、风险提示", font, h2))
    story.append(
        _p(
            "指数编制与可交易组合存在差异；与国债期货久期、杠杆不一致。仅供研究整理，不构成投资建议。",
            font,
            body,
        )
    )

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build(story)


def main() -> None:
    if not FACTORS_CSV.exists():
        raise FileNotFoundError(f"缺少 {FACTORS_CSV}，请先运行 reproduce/build_public_macro_dataset.py")

    df = pd.read_csv(FACTORS_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    for c in (COL_LONG, COL_SHORT, COL_LS):
        if c not in df.columns:
            raise KeyError(
                f"因子表缺少列 {c}。请重新运行 build_public_macro_dataset.py（已改为国债 CSI931018 + 利差列）。"
            )

    nav_long = level_to_cumnav(df[COL_LONG])
    nav_short = level_to_cumnav(df[COL_SHORT])
    nav_ls = pd.to_numeric(df[COL_LS], errors="coerce")
    rl = pd.to_numeric(df[COL_LONG], errors="coerce").pct_change()
    rs = pd.to_numeric(df[COL_SHORT], errors="coerce").pct_change()
    ls_ret = rl - rs

    # 周频压缩后再评估
    wk = pd.DataFrame({"Date": df["Date"], "r_long": rl, "r_short": rs, "r_ls": ls_ret})
    wk = wk.set_index("Date")
    wk["r_long"] = (1.0 + wk["r_long"]).resample("W-FRI").prod() - 1.0
    wk["r_short"] = (1.0 + wk["r_short"]).resample("W-FRI").prod() - 1.0
    wk["r_ls"] = (1.0 + wk["r_ls"]).resample("W-FRI").prod() - 1.0
    wk = wk.dropna(how="all")
    wk = wk.reset_index()

    df = wk.copy()
    rl = df["r_long"]
    rs = df["r_short"]
    ls_ret = df["r_ls"]
    nav_long = (1.0 + rl).cumprod()
    nav_short = (1.0 + rs).cumprod()
    nav_ls = (1.0 + ls_ret).cumprod()
    years = sample_years(df["Date"])

    metrics = {
        "years": years,
        "c_long": cagr_from_nav(nav_long, years),
        "c_short": cagr_from_nav(nav_short, years),
        "c_ls": cagr_from_nav(nav_ls, years),
        "vol_ls": ann_vol_daily(ls_ret),
        "mdd_long": max_drawdown(nav_long),
        "mdd_short": max_drawdown(nav_short),
        "mdd_ls": max_drawdown(nav_ls),
        "corr": float(rl.corr(rs)) if rl.notna().any() else float("nan"),
    }

    nav_hedge: pd.Series | None = None
    if COL_HEDGE in pd.read_csv(FACTORS_CSV).columns:
        nav_hedge_daily = pd.to_numeric(pd.read_csv(FACTORS_CSV)[COL_HEDGE], errors="coerce")
        df_daily_dates = pd.to_datetime(pd.read_csv(FACTORS_CSV)["Date"])
        hedge_wk = pd.DataFrame({"Date": df_daily_dates, "HedgeRet": nav_hedge_daily.pct_change()}).set_index("Date")
        hedge_ret = (1.0 + hedge_wk["HedgeRet"]).resample("W-FRI").prod() - 1.0
        nav_hedge = (1.0 + hedge_ret.fillna(0.0)).cumprod()
        nav_hedge = nav_hedge.reindex(pd.DatetimeIndex(df["Date"])).ffill()
        metrics["c_hedge"] = cagr_from_nav(nav_hedge, years)
        metrics["mdd_hedge"] = max_drawdown(nav_hedge)
        metrics["vol_hedge"] = ann_vol_daily(hedge_ret)

    base_daily = pd.read_csv(FACTORS_CSV)
    base_daily["Date"] = pd.to_datetime(base_daily["Date"])
    for c in (COL_STRAT_LO,):
        if c not in base_daily.columns:
            raise KeyError(f"因子表缺少列 {c}，请运行 reproduce/build_public_macro_dataset.py 以生成 LongOnly 策略列。")
    nav_strat_lo_daily = pd.to_numeric(base_daily[COL_STRAT_LO], errors="coerce")
    lo_wk_ret = pd.DataFrame({"Date": base_daily["Date"], "r": nav_strat_lo_daily.pct_change()}).set_index("Date")
    lo_wk_ret = (1.0 + lo_wk_ret["r"]).resample("W-FRI").prod() - 1.0
    nav_strat_lo = (1.0 + lo_wk_ret.fillna(0.0)).cumprod()
    nav_strat_lo = nav_strat_lo.reindex(pd.DatetimeIndex(df["Date"])).ffill()
    nav_strat_sw: pd.Series | None = None
    if COL_STRAT_SW in base_daily.columns:
        nav_strat_sw_daily = pd.to_numeric(base_daily[COL_STRAT_SW], errors="coerce")
        sw_wk_ret = pd.DataFrame({"Date": base_daily["Date"], "r": nav_strat_sw_daily.pct_change()}).set_index("Date")
        sw_wk_ret = (1.0 + sw_wk_ret["r"]).resample("W-FRI").prod() - 1.0
        nav_strat_sw = (1.0 + sw_wk_ret.fillna(0.0)).cumprod()
        nav_strat_sw = nav_strat_sw.reindex(pd.DatetimeIndex(df["Date"])).ffill()
    metrics["strat_lo_cagr"] = cagr_from_nav(nav_strat_lo, years)
    metrics["strat_lo_vol"] = ann_vol_daily(rl)
    metrics["strat_lo_mdd"] = max_drawdown(nav_strat_lo)
    # LongShort 仍作为“利差口径”在二、指标部分展示（c_ls / vol_ls / mdd_ls），但不再作为“最终策略”展示项
    if nav_strat_sw is not None and nav_strat_sw.notna().any():
        sw_ret = nav_strat_sw.pct_change()
        metrics["strat_sw_cagr"] = cagr_from_nav(nav_strat_sw, years)
        metrics["strat_sw_vol"] = ann_vol_daily(sw_ret)
        metrics["strat_sw_mdd"] = max_drawdown(nav_strat_sw)
        if "5_pos_switch_gov_weight" in base_daily.columns and "5_pos_switch_cp_weight" in base_daily.columns:
            wg = pd.to_numeric(base_daily["5_pos_switch_gov_weight"], errors="coerce").fillna(0.0)
            wc = pd.to_numeric(base_daily["5_pos_switch_cp_weight"], errors="coerce").fillna(0.0)
            traded = ((wg.diff().abs() + wc.diff().abs()) > 1e-12).sum()
            metrics["strat_sw_trades"] = float(traded)
        metrics["strat_sw_slip"] = 0.0001

    rl_d = pd.to_numeric(base_daily[COL_LONG], errors="coerce").pct_change()
    rs_d = pd.to_numeric(base_daily[COL_SHORT], errors="coerce").pct_change()
    tw_lo = longonly_buyhold_win_rate(rl_d)
    metrics["strat_wr_lo_long"] = fmt_pct(tw_lo["timing_long_win_rate"])
    metrics["strat_wr_lo_short"] = "—"
    if "5_pos_switch_gov_weight" in base_daily.columns and "5_pos_switch_cp_weight" in base_daily.columns:
        tw_ls = timing_win_rates_gov_cp(
            rl_d,
            rs_d,
            pd.to_numeric(base_daily["5_pos_switch_gov_weight"], errors="coerce").fillna(0.0),
            pd.to_numeric(base_daily["5_pos_switch_cp_weight"], errors="coerce").fillna(0.0),
        )
        metrics["strat_wr_ls_long"] = fmt_pct(tw_ls["timing_long_win_rate"])
        metrics["strat_wr_ls_short"] = fmt_pct(tw_ls["timing_short_win_rate"])
    else:
        metrics["strat_wr_ls_long"] = "—"
        metrics["strat_wr_ls_short"] = "—"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_strategies_figure(df, nav_strat_lo, nav_strat_sw, metrics)
    plot_figure(df, nav_long, nav_short, nav_ls, metrics, nav_hedge)

    pd.DataFrame(
        {
            "Date": df["Date"],
            "LongOnly_nav": nav_strat_lo.values,
            "LongShort_nav": nav_strat_sw.values if nav_strat_sw is not None else np.nan,
        }
    ).to_csv(OUT_STRAT_CSV, index=False, encoding="utf-8-sig")

    html_doc = build_html(df, metrics)
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    build_pdf(df, metrics)

    # 便于与 111.htm 并列打开：复制主题文件（可选阅读）
    themed = ROOT / "111.files" / "themedata.thmx"
    if themed.exists():
        shutil.copy2(themed, OUT_DIR / "themedata_ref.thmx")

    print("saved", OUT_HTML)
    print("saved", OUT_PDF)
    print("saved", OUT_FIG)
    print("saved", OUT_FIG_STRAT)
    print("saved", OUT_STRAT_CSV)
    msg = (
        f"CAGR long={metrics['c_long']*100:.3f}% short={metrics['c_short']*100:.3f}% "
        f"1:1_LS={metrics['c_ls']*100:.3f}% vol_ls={metrics['vol_ls']*100:.3f}%"
    )
    if "c_hedge" in metrics:
        msg += f"  beta_hedge={metrics['c_hedge']*100:.3f}% vol={metrics['vol_hedge']*100:.3f}%"
    msg += (
        f"  LongOnly={metrics['strat_lo_cagr']*100:.3f}%"
    )
    if "strat_sw_cagr" in metrics:
        msg += f" LongShort={metrics['strat_sw_cagr']*100:.3f}%"
    print(msg)


if __name__ == "__main__":
    main()
