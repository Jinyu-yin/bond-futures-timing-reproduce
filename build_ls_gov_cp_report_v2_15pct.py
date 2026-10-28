"""
Report v2 (15% 版本): 国债活跃券 vs 短融 LongShort（切换版）
----------------------------------------------------------
与 `build_ls_gov_cp_report_v2.py` 的 SOP 版不同：
- 因子筛选：按“原有逻辑”做分组动量优选，但将“固定 topN”替换为“每组取前 15%”。
- 信号与回测输入：使用根目录 `input_data.xlsx` 的 Price/Factors。
- 交易标的：使用 `reproduce/public_factors.csv` 的国债(931018)与短融(H11014)序列来计算 LongOnly 与 LongShort 净值。
- 输出目录：`reproduce/reports/gov_cp_ls/report_v2_15pct/`
"""

from __future__ import annotations

import html
import shutil
from dataclasses import dataclass
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

ROOT = Path(__file__).resolve().parents[1]

INPUT_XLSX = ROOT / "input_data.xlsx"
PUBLIC_FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"

OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls" / "report_v2_15pct"
OUT_HTML = OUT_DIR / "report.html"
OUT_PDF = OUT_DIR / "report.pdf"
OUT_FIG_STRAT = OUT_DIR / "fig_strategies_longonly_longshort.png"
OUT_STRAT_CSV = OUT_DIR / "strategies_nav.csv"
OUT_SEL_LOG = OUT_DIR / "selection_log.csv"

COL_GOV = "5_long_gov_510y_csi931018_close"
COL_CP = "5_short_cp_csi_close"


def _pdf_font() -> str:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "STSong-Light"


def _p(text: str, st: ParagraphStyle) -> Paragraph:
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


def ts_zscore(s: pd.Series, win: int, minp: int) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std().replace(0, np.nan)
    return (s - mu) / sd


@dataclass(frozen=True)
class Cfg15:
    start: str = "2019-01-01"
    end: str = "2025-12-31"

    # factor signal construction (same as paper_style_bond_futures_timing)
    z_win_level: int = 252
    z_minp_level: int = 60
    z_win_diff: int = 63
    z_minp_diff: int = 20

    # momentum scoring windows
    mom_short: int = 10
    mom_long: int = 252

    # monthly rebalance
    rebalance: str = "M"  # monthly first trading day

    # selection rule: top 15% per group
    top_pct: float = 0.15

    # trading mapping to gov/cp
    slip_per_turn: float = 0.0001  # 0.01% per trade-event
    open_threshold: int = 0  # same meaning as paper_style (0 => sign, 1 => require |sum|>1)


def build_single_factor_signal(x: pd.Series, cfg: Cfg15) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    strength = ts_zscore(x, win=cfg.z_win_level, minp=cfg.z_minp_level)
    change = ts_zscore(x.diff(1), win=cfg.z_win_diff, minp=cfg.z_minp_diff)
    cond = (change > 0) | (strength > 0)
    sig = pd.Series(np.where(cond, -1.0, 1.0), index=x.index, dtype=float)
    sig = sig.where(x.notna(), 0.0)
    return sig.fillna(0.0)


def month_first_trading_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=idx)
    first = s.groupby(s.index.to_period("M")).apply(lambda x: x.index.min())
    return pd.DatetimeIndex(first.values)


def momentum_return(nav: pd.Series, win: int, at: int) -> float:
    if at - win < 0:
        return float("nan")
    a = nav.iloc[at]
    b = nav.iloc[at - win]
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return float(a / b - 1.0)


def composite_rank(mom_s: pd.Series, mom_l: pd.Series) -> pd.Series:
    r1 = mom_s.rank(ascending=False, method="average")
    r2 = mom_l.rank(ascending=False, method="average")
    return (r1 + r2) / 2.0


def build_report_html(metrics: dict[str, str], strat_table_html: str) -> str:
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td style='text-align:right'>{html.escape(str(v))}</td></tr>"
        for k, v in metrics.items()
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>国债活跃券与短融 LongShort 简报（v2-15%）</title>
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
  <h1>国债活跃券与短融：LongShort 简报（v2-15%）</h1>
  <p class="muted">生成时间：{html.escape(gen)}　｜　信号输入：<code>input_data.xlsx</code>　｜　标的价格：<code>reproduce/public_factors.csv</code></p>

  <h2>一、方法说明（v2-15%）</h2>
  <p>本版使用“原有分组动量优选”框架，但将每组固定 topN 替换为<strong>取前 15%</strong>（按单因子净值动量打分排序后，按比例截取）。</p>

  <h2>二、样本区间与指标</h2>
  <table class="stats">
    <tr><th>指标</th><th>数值</th></tr>
    {rows}
  </table>

  <h2>三、最终策略：LongOnly 与 LongShort</h2>
  {strat_table_html}
  <div class="fig"><img src="{OUT_FIG_STRAT.name}" alt="LongOnly vs LongShort" /></div>

  <h2>四、风险提示</h2>
  <p>仅供研究整理，不构成投资建议。</p>
</div>
</body>
</html>
"""


def build_pdf(metrics_rows: list[list[str]], strat_rows: list[list[str]]) -> None:
    font = _pdf_font()
    title = ParagraphStyle("t", fontName=font, fontSize=18, leading=24, textColor=colors.HexColor("#0F4761"), spaceAfter=12)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#0F4761"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", fontName=font, fontSize=10.5, leading=17, spaceAfter=6, wordWrap="CJK")
    small = ParagraphStyle("s", fontName=font, fontSize=9, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=8)

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    story: list = []
    story.append(_p("国债活跃券与短融：LongShort 简报（v2-15%）", title))
    story.append(_p(f"生成时间：{gen}　信号输入：input_data.xlsx　标的价格：reproduce/public_factors.csv", small))

    story.append(_p("一、方法说明（v2-15%）", h2))
    story.append(_p("分组动量优选框架；每组取单因子动量打分的前15%进入组合。", body))

    story.append(_p("二、样本区间与指标", h2))
    tbl_data = [[_p("指标", body), _p("数值", body)]] + [[_p(a, body), _p(b, body)] for a, b in metrics_rows]
    tbl = Table(tbl_data, colWidths=[8.5 * cm, 8.5 * cm])
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb3c8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8"))]))
    story.append(tbl)
    story.append(Spacer(1, 8))

    story.append(_p("三、最终策略 LongOnly 与 LongShort", h2))
    st_hdr = ["策略", "口径", "近似年化", "年化波动", "最大回撤"]
    st_data = [[_p(x, body) for x in st_hdr]] + [[_p(x, body) for x in row] for row in strat_rows]
    st_tbl = Table(st_data, colWidths=[2.4 * cm, 6.2 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])
    st_tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb3c8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8"))]))
    story.append(st_tbl)
    story.append(Spacer(1, 6))

    if OUT_FIG_STRAT.exists():
        story.append(_fit_img(Image(str(OUT_FIG_STRAT)), max_w=16 * cm, max_h=9.5 * cm))

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    doc.build(story)


def plot_strat(dates: pd.Series, nav_lo: pd.Series, nav_ls: pd.Series, cagr_lo: float, cagr_ls: float) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(dates, nav_lo, lw=2.2, color="#1f77b4", label=f"LongOnly（Gov） CAGR≈{cagr_lo*100:.2f}%")
    ax.plot(dates, nav_ls, lw=2.2, color="#2ca02c", label=f"LongShort（15%筛选） CAGR≈{cagr_ls*100:.2f}%")
    ax.set_title("策略净值对比（起点=1）｜LongOnly vs LongShort（15%筛选信号）")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_STRAT, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Cfg15()

    if not INPUT_XLSX.exists():
        raise FileNotFoundError(INPUT_XLSX)
    if not PUBLIC_FACTORS_CSV.exists():
        raise FileNotFoundError(PUBLIC_FACTORS_CSV)

    # 1) Load input_data.xlsx
    price_df = pd.read_excel(INPUT_XLSX, sheet_name="Price")
    fac_df = pd.read_excel(INPUT_XLSX, sheet_name="Factors")
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    fac_df["Date"] = pd.to_datetime(fac_df["Date"])
    px = price_df.sort_values("Date").set_index("Date")["Close"].astype(float)
    ret = px.pct_change().fillna(0.0)
    f = fac_df.sort_values("Date").set_index("Date").select_dtypes(include=["number"]).reindex(px.index).ffill(limit=30)

    # split groups by prefix
    cols = [c for c in f.columns if str(c).strip()]
    gcols = {
        "1_growth": [c for c in cols if str(c).startswith("1_")],
        "2_inflation": [c for c in cols if str(c).startswith("2_")],
        "3_credit": [c for c in cols if str(c).startswith("3_")],
        "4_omo": [c for c in cols if str(c).startswith("4_")],
    }

    # single factor signals + nav
    sig = pd.concat({c: build_single_factor_signal(f[c], cfg) for c in cols}, axis=1).fillna(0.0)
    pnl = sig.shift(1).mul(ret, axis=0).fillna(0.0)
    nav_factor = (1.0 + pnl).cumprod()

    # rebalance monthly first trading day
    reb = month_first_trading_days(px.index)
    group_sig = pd.DataFrame(0.0, index=px.index, columns=list(gcols.keys()))
    log_rows: list[dict] = []

    dates = px.index
    for i, d in enumerate(reb):
        d_next = reb[i + 1] if i + 1 < len(reb) else dates[-1] + pd.Timedelta(days=1)
        mask = (dates >= d) & (dates < d_next)
        at = int(dates.get_loc(d))
        last = at - 1
        if last < max(cfg.mom_long, cfg.mom_short, 5):
            group_sig.loc[mask, :] = 0.0
            continue

        sel_counts: dict[str, int] = {}
        for g, gc in gcols.items():
            if not gc:
                group_sig.loc[mask, g] = 0.0
                sel_counts[g] = 0
                continue

            mom_s = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_short, last) for c in gc})
            mom_l = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_long, last) for c in gc})
            cr = composite_rank(mom_s, mom_l).dropna()
            # keep non-negative long momentum (same as original)
            cr = cr[[c for c in cr.index if pd.notna(mom_l.get(c)) and float(mom_l[c]) >= 0.0]]
            if cr.empty:
                group_sig.loc[mask, g] = 0.0
                sel_counts[g] = 0
                continue

            n = max(1, int(np.ceil(cfg.top_pct * len(cr))))
            chosen = cr.sort_values().head(n).index.tolist()
            sel_counts[g] = len(chosen)

            ssum = sig.loc[mask, chosen].sum(axis=1)
            gs = pd.Series(0.0, index=dates[mask])
            gs[ssum > 0] = 1.0
            gs[ssum < 0] = -1.0
            group_sig.loc[mask, g] = gs

        log_rows.append({"Date": d.date().isoformat(), **{f"sel_{k}": v for k, v in sel_counts.items()}})

    gsum = group_sig.sum(axis=1)
    final_pos = pd.Series(0.0, index=px.index)
    if cfg.open_threshold <= 0:
        final_pos[gsum > 0] = 1.0
        final_pos[gsum < 0] = -1.0
    else:
        final_pos[gsum > cfg.open_threshold] = 1.0
        final_pos[gsum < -cfg.open_threshold] = -1.0

    # 2) Map to gov/cp longshort and compute nav using public_factors.csv
    pub = pd.read_csv(PUBLIC_FACTORS_CSV)
    pub["Date"] = pd.to_datetime(pub["Date"])
    pub = pub.sort_values("Date").set_index("Date")
    for c in (COL_GOV, COL_CP):
        if c not in pub.columns:
            raise KeyError(f"Missing {c} in {PUBLIC_FACTORS_CSV}")
    gov = pd.to_numeric(pub[COL_GOV], errors="coerce")
    cp = pd.to_numeric(pub[COL_CP], errors="coerce")
    rl = gov.pct_change().fillna(0.0)
    rs = cp.pct_change().fillna(0.0)

    common = pub.index.intersection(final_pos.index)
    pos = final_pos.reindex(common).fillna(0.0)
    pos_exe = pos.shift(1).fillna(0.0)
    r_g = rl.reindex(common).fillna(0.0)
    r_c = rs.reindex(common).fillna(0.0)
    strat_ret = pd.Series(0.0, index=common)
    strat_ret[pos_exe > 0] = r_g[pos_exe > 0]
    strat_ret[pos_exe < 0] = r_c[pos_exe < 0]
    turn = pos_exe.diff().abs().fillna(0.0)
    strat_ret = strat_ret - cfg.slip_per_turn * (turn > 1e-12).astype(float)

    nav_lo = (1.0 + r_g).cumprod()
    nav_ls = (1.0 + strat_ret).cumprod()

    years = sample_years(pd.Series(common))
    c_lo = cagr_from_nav(nav_lo, years)
    c_ls = cagr_from_nav(nav_ls, years)
    metrics = {
        "样本起止": f"{common.min().date()} ~ {common.max().date()}",
        "样本跨度（年）": f"{years:.3f}",
        "LongOnly 近似年化": f"{c_lo*100:.3f}%",
        "LongShort 近似年化": f"{c_ls*100:.3f}%",
        "LongOnly 年化波动": f"{ann_vol_daily(r_g)*100:.3f}%",
        "LongShort 年化波动": f"{ann_vol_daily(strat_ret)*100:.3f}%",
        "LongOnly 最大回撤": f"{max_drawdown(nav_lo)*100:.2f}%",
        "LongShort 最大回撤": f"{max_drawdown(nav_ls)*100:.2f}%",
        "换仓次数（估计）": f"{int((turn > 1e-12).sum())}",
        "滑点（每次）": f"{cfg.slip_per_turn*100:.2f}%",
        "每组取前比例": f"{cfg.top_pct*100:.1f}%",
    }

    # outputs
    pd.DataFrame(log_rows).to_csv(OUT_SEL_LOG, index=False, encoding="utf-8-sig")
    pd.DataFrame({"Date": common, "LongOnly_nav": nav_lo.values, "LongShort_nav": nav_ls.values}).to_csv(OUT_STRAT_CSV, index=False, encoding="utf-8-sig")
    plot_strat(pd.Series(common), nav_lo, nav_ls, c_lo, c_ls)

    strat_rows = [
        ["LongOnly", "买入并持有国债活跃券指数（931018）", metrics["LongOnly 近似年化"], metrics["LongOnly 年化波动"], metrics["LongOnly 最大回撤"]],
        ["LongShort", "15%筛选信号驱动：多国债/多短融二选一切换", metrics["LongShort 近似年化"], metrics["LongShort 年化波动"], metrics["LongShort 最大回撤"]],
    ]
    strat_table_html = f"""
  <p class="muted">信号：分组动量优选→每组取前 {metrics['每组取前比例']}；换仓次数≈{metrics['换仓次数（估计）']}；滑点 {metrics['滑点（每次）']}。</p>
  <table class="stats">
    <tr><th>策略</th><th>口径</th><th>近似年化</th><th>年化波动</th><th>最大回撤</th></tr>
    <tr><td>{strat_rows[0][0]}</td><td>{strat_rows[0][1]}</td><td style="text-align:right">{strat_rows[0][2]}</td><td style="text-align:right">{strat_rows[0][3]}</td><td style="text-align:right">{strat_rows[0][4]}</td></tr>
    <tr><td>{strat_rows[1][0]}</td><td>{strat_rows[1][1]}</td><td style="text-align:right">{strat_rows[1][2]}</td><td style="text-align:right">{strat_rows[1][3]}</td><td style="text-align:right">{strat_rows[1][4]}</td></tr>
  </table>
"""

    html_doc = build_report_html(metrics, strat_table_html)
    OUT_HTML.write_text(html_doc, encoding="utf-8")

    build_pdf([[k, v] for k, v in metrics.items()], strat_rows)

    themed = ROOT / "111.files" / "themedata.thmx"
    if themed.exists():
        shutil.copy2(themed, OUT_DIR / "themedata_ref.thmx")

    print("saved", OUT_HTML)
    print("saved", OUT_PDF)
    print("saved", OUT_FIG_STRAT)
    print("saved", OUT_STRAT_CSV)
    print("saved", OUT_SEL_LOG)


if __name__ == "__main__":
    main()

