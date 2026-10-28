"""
Report v2: 国债活跃券 vs 短融 LongShort（切换版）
-------------------------------------------------
相较 v1：
- 因子筛选与合成信号：仅按训练期收益率筛选，并做相关性去冗余；合成信号输出 Static/Dynamic
- 输入数据：充分使用根目录 `input_data.xlsx` 的 Price/Factors（用于筛选与信号）
- 交易标的：仍使用 `reproduce/public_factors.csv` 的国债(931018)与短融(H11014)序列来计算 LongOnly 与 LongShort 净值
- 输出落地到 `reproduce/reports/gov_cp_ls/report_v2/`，不覆盖 v1
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

from gov_cp_timing_metrics import fmt_pct, longonly_buyhold_win_rate, timing_win_rates_from_position

ROOT = Path(__file__).resolve().parents[1]

# inputs
INPUT_XLSX = ROOT / "input_data.xlsx"
PUBLIC_FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"

# outputs
OUT_DIR = ROOT / "reproduce" / "reports" / "gov_cp_ls" / "report_v2"
OUT_HTML = OUT_DIR / "report.html"
OUT_PDF = OUT_DIR / "report.pdf"
OUT_FIG_STRAT = OUT_DIR / "fig_strategies_longonly_longshort.png"
OUT_STRAT_CSV = OUT_DIR / "strategies_nav.csv"
OUT_SEL_LOG = OUT_DIR / "selection_log.csv"

# public series columns
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


@dataclass(frozen=True)
class SOPConfig:
    start_date: str = "2019-01-01"
    end_date: str = "2025-12-31"
    lookback_years: int = 3
    rebalance_months: int = 12

    horizons: tuple[int, ...] = (1, 5, 10, 22)
    corr_threshold: float = 0.75

    dyn_ic_window: int = 60
    penalty_factor: float = 0.0
    max_selected_factors: int = 20

    # trading for gov/cp switch
    slip_per_turn: float = 0.0001  # 0.01% per side proxy per "trade event" (daily)
    pos_threshold: float = 0.5  # discrete position threshold on composite signal


@dataclass(frozen=True)
class HMMConfig:
    enabled: bool = False
    n_states: int = 2
    fit_window: int = 504
    min_history: int = 252
    refit_freq: str = "ME"
    vol_window: int = 20
    crowd_window: int = 20
    shock_window: int = 5
    bias_strength: float = 0.35
    scale_strength: float = 0.30
    min_scale: float = 0.75
    max_scale: float = 1.15
    covariance_type: str = "diag"
    n_iter: int = 200
    random_state: int = 42


class SOPSelector:
    def __init__(self, price: pd.Series, factors: pd.DataFrame, cfg: SOPConfig):
        self.cfg = cfg
        self.price = price.sort_index()
        self.ret_daily = self.price.pct_change().shift(-1)
        self.df_factors = factors.sort_index()

    def get_stress_period_return(self, signal: pd.Series, ret: pd.Series) -> float:
        try:
            rolling_ret = ret.rolling(90).sum()
            worst_date = rolling_ret.idxmin()
            if pd.isna(worst_date):
                return 0.0
            start_stress = worst_date - pd.Timedelta(days=90)
            end_stress = worst_date
            sub_sig = signal.loc[start_stress:end_stress]
            sub_ret = ret.loc[start_stress:end_stress]
            if len(sub_sig) < 10:
                return 0.0
            return float((sub_sig * sub_ret).sum())
        except Exception:
            return 0.0

    def calculate_single_factor_metrics(self, name: str, f_series: pd.Series, p_series: pd.Series) -> dict | None:
        common = f_series.dropna().index.intersection(p_series.index)
        f = pd.to_numeric(f_series.loc[common], errors="coerce")
        p = pd.to_numeric(p_series.loc[common], errors="coerce")
        if f.notna().sum() < 250:
            return None

        best_ic, best_h = 0.0, 1
        for h in self.cfg.horizons:
            r_h = p.pct_change(h).shift(-h)
            valid = f.index.intersection(r_h.dropna().index)
            if len(valid) < 60:
                continue
            curr_ic = float(f.loc[valid].corr(r_h.loc[valid]))
            if abs(curr_ic) > abs(best_ic):
                best_ic = curr_ic
                best_h = int(h)

        r_daily = p.pct_change().shift(-1)
        valid_idx = f.index.intersection(r_daily.dropna().index)
        f_final = f.loc[valid_idx].dropna()
        r_final = r_daily.loc[f_final.index].dropna()
        if len(f_final) < 250:
            return None

        f_final = f_final.clip(f_final.quantile(0.01), f_final.quantile(0.99))
        f_demean = f_final - f_final.rolling(250, min_periods=60).mean()
        f_signal = np.sign(f_demean).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        direction = -1.0 if best_ic < 0 else 1.0
        train_ret = (f_signal * r_final * direction).dropna()
        train_nav = (1.0 + train_ret).cumprod()
        total_return = float(train_nav.iloc[-1] - 1.0) if len(train_nav) else float("nan")

        return {
            "Factor": name,
            "Best_Horizon": best_h,
            "Best_IC": best_ic,
            "Direction": direction,
            "Train_Total_Return": total_return,
        }

    def screen_factors(self, train_start: pd.Timestamp, train_end: pd.Timestamp) -> pd.DataFrame:
        f_train = self.df_factors.loc[train_start:train_end]
        p_train = self.price.loc[train_start:train_end]
        if len(f_train) < 250:
            return pd.DataFrame()

        metrics_list: list[dict] = []
        for col in f_train.columns:
            if str(col).strip().lower() == "date":
                continue
            try:
                res = self.calculate_single_factor_metrics(str(col), f_train[col], p_train)
                if res is not None:
                    metrics_list.append(res)
            except Exception:
                continue

        if not metrics_list:
            return pd.DataFrame()
        df_metrics = pd.DataFrame(metrics_list)

        df_metrics["Train_Total_Return"] = pd.to_numeric(df_metrics["Train_Total_Return"], errors="coerce")
        df_passed = df_metrics.dropna(subset=["Train_Total_Return"]).copy()
        df_passed = df_passed[df_passed["Train_Total_Return"] > 0]
        if df_passed.empty:
            return pd.DataFrame()
        df_passed = df_passed.sort_values("Train_Total_Return", ascending=False)

        pool_cols = df_passed["Factor"].tolist()
        pool_data = f_train[pool_cols].copy()
        corr_mat = pool_data.corr().abs()

        final: list[pd.Series] = []
        selected_names: list[str] = []
        for _, r in df_passed.iterrows():
            name = str(r["Factor"])
            if name not in corr_mat.index:
                continue
            is_high_corr = any(float(corr_mat.loc[name, ex]) > self.cfg.corr_threshold for ex in selected_names)
            if not is_high_corr:
                final.append(r)
                selected_names.append(name)

        out = pd.DataFrame(final).head(self.cfg.max_selected_factors)
        return out.reset_index(drop=True)

    def generate_signals(self, factor_data: pd.DataFrame, factor_config: pd.DataFrame) -> pd.DataFrame:
        if factor_config is None or factor_config.empty or factor_data is None or factor_data.empty:
            return pd.DataFrame(0.0, index=factor_data.index if factor_data is not None else None, columns=["Static", "Dynamic"])

        raw_signals = pd.DataFrame(index=factor_data.index)
        dynamic_weights = pd.DataFrame(index=factor_data.index)
        for _, row in factor_config.iterrows():
            name = str(row["Factor"])
            direction = float(row["Direction"])
            horizon = int(row["Best_Horizon"])
            if name not in factor_data.columns:
                continue
            raw = pd.to_numeric(factor_data[name], errors="coerce")
            roll_mean = raw.rolling(250, min_periods=60).mean()
            roll_std = raw.rolling(250, min_periods=60).std().replace(0, np.nan).ffill().fillna(1.0)
            z = ((raw - roll_mean) / (roll_std + 1e-6)).clip(-4.0, 4.0)
            if horizon > 1:
                z = z.rolling(horizon).mean()
            raw_signals[name] = z * direction
            dynamic_weights[name] = 1.0 if float(row["Best_IC"]) >= 0 else 0.0

        if raw_signals.empty:
            return pd.DataFrame(0.0, index=factor_data.index, columns=["Static", "Dynamic"])

        comp_static = raw_signals.mean(axis=1)

        try:
            weighted = raw_signals.mul(dynamic_weights, axis=1)
            weight_sum = dynamic_weights.sum(axis=1).replace(0, np.nan)
            comp_dynamic = (weighted.sum(axis=1) / weight_sum).fillna(comp_static)
        except Exception:
            comp_dynamic = comp_static

        df_final = pd.DataFrame({"Static": comp_static, "Dynamic": comp_dynamic})
        for col in df_final.columns:
            vol_scale = df_final[col].rolling(60).std()
            safe_vol = vol_scale.clip(lower=0.2)
            df_final[col] = (df_final[col] / (safe_vol + 1e-6)).fillna(0.0).clip(-3.0, 3.0)
        return df_final


def build_report_html(df_metrics: dict, strat_table_html: str) -> str:
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td style='text-align:right'>{html.escape(str(v))}</td></tr>"
        for k, v in df_metrics.items()
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>国债活跃券与短融 LongShort 简报（v2）</title>
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
  <h1>国债活跃券与短融：LongShort 简报（v2）</h1>
  <p class="muted">生成时间：{html.escape(gen)}　｜　信号输入：<code>input_data.xlsx</code>　｜　标的价格：<code>reproduce/public_factors.csv</code></p>

  <h2>一、方法说明（v2）</h2>
  <p>本版沿用原有 V2 的基本框架：因子筛选只看训练期累计收益率（Train_Total_Return），先保留收益为正且更高的因子，再按相关性阈值去冗余；合成信号包含静态等权（Static）与动态加权（Dynamic）。</p>
  <p>相较更早版本，本版主要在信号生成与仓位映射上做了稳健化：因子统一做滚动标准化、按持有期平滑，再做波动率缩放与限幅，随后映射为国债/短融的离散切换仓位，以降低单日极端信号对结果的影响。</p>

  <h2>二、样本区间与指标</h2>
  <table class="stats">
    <tr><th>指标</th><th>数值</th></tr>
    {rows}
  </table>

  <h2>三、最终策略：LongOnly 与 LongShort</h2>
  {strat_table_html}
  <div class="fig"><img src="{OUT_FIG_STRAT.name}" alt="LongOnly vs LongShort" /></div>

  <h2>四、结果解读</h2>
  <p><b>LongOnly</b> 代表一直持有国债活跃券指数，可视作防守基准；<b>LongShort</b> 则代表在国债与短融之间按动态信号切换，若动态信号转弱则更倾向退出或切换到短融。</p>
  <p>若 LongShort 的 Sharpe、回撤控制和胜率均优于 LongOnly，说明该轮筛选与动态加权在样本内更有效；反之则说明当前因子池或状态环境更适合简单持有。</p>

  <h2>五、风险提示</h2>
  <p>本结果基于公开/整理数据与简化交易成本口径，仍不等同于可交易组合。仅供研究整理，不构成投资建议。</p>
</div>
</body>
</html>
"""


def build_pdf(html_metrics_rows: list[list[str]], strat_rows: list[list[str]]) -> None:
    font = _pdf_font()
    title = ParagraphStyle("t", fontName=font, fontSize=18, leading=24, textColor=colors.HexColor("#0F4761"), spaceAfter=12)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=13, leading=18, textColor=colors.HexColor("#0F4761"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("b", fontName=font, fontSize=10.5, leading=17, spaceAfter=6, wordWrap="CJK")
    small = ParagraphStyle("s", fontName=font, fontSize=9, leading=14, textColor=colors.HexColor("#555555"), spaceAfter=8)

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    story: list = []
    story.append(_p("国债活跃券与短融：LongShort 简报（v2）", title))
    story.append(_p(f"生成时间：{gen}　信号输入：input_data.xlsx　标的价格：reproduce/public_factors.csv", small))

    story.append(_p("一、方法说明（v2）", h2))
    story.append(_p("因子筛选仅基于训练期累计收益率（Train_Total_Return），并做相关性去冗余；输出 Static/Dynamic 合成信号。", body))

    story.append(_p("二、样本区间与指标", h2))
    tbl_data = [[_p("指标", body), _p("数值", body)]] + [[_p(a, body), _p(b, body)] for a, b in html_metrics_rows]
    tbl = Table(tbl_data, colWidths=[8.5 * cm, 8.5 * cm])
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
    story.append(Spacer(1, 8))

    story.append(_p("三、最终策略 LongOnly 与 LongShort", h2))
    st_hdr = ["策略", "口径", "近似年化", "年化波动", "最大回撤", "多头胜率", "空头胜率"]
    st_data = [[_p(x, body) for x in st_hdr]] + [[_p(x, body) for x in row] for row in strat_rows]
    st_tbl = Table(st_data, colWidths=[2.0 * cm, 4.8 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 1.9 * cm, 1.9 * cm])
    st_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fb3c8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(st_tbl)
    story.append(Spacer(1, 6))

    if OUT_FIG_STRAT.exists():
        story.append(_fit_img(Image(str(OUT_FIG_STRAT)), max_w=16 * cm, max_h=9.5 * cm))

    story.append(Spacer(1, 8))
    story.append(_p("四、风险提示", h2))
    story.append(_p("仅供研究整理，不构成投资建议。", body))

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build(story)


def plot_strat(df: pd.DataFrame, nav_lo: pd.Series, nav_ls: pd.Series, metrics: dict) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 5.2))
    clo = metrics.get("cagr_lo", float("nan"))
    cls = metrics.get("cagr_ls", float("nan"))
    l1 = f"LongOnly（Gov） CAGR≈{clo * 100:.2f}%" if clo == clo and np.isfinite(clo) else "LongOnly（Gov）"
    l2 = f"LongShort（Gov/CP切换） CAGR≈{cls * 100:.2f}%" if cls == cls and np.isfinite(cls) else "LongShort（Gov/CP切换）"
    ax.plot(df["Date"], nav_lo, lw=2.2, color="#1f77b4", label=l1)
    ax.plot(df["Date"], nav_ls, lw=2.2, color="#2ca02c", label=l2)
    ax.set_title("策略净值对比（起点=1）｜LongOnly vs LongShort（周频）")
    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.02,
        "LongShort：信号来自 input_data.xlsx 的因子训练收益率筛选（Dynamic-Discrete），"
        "仓位>0.5 做多国债、<-0.5 做多短融，换仓扣固定滑点。",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="#f5f5dc", alpha=0.45),
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_STRAT, dpi=150)
    plt.close(fig)


def compute_v2_daily_strategy_panel(
    input_xlsx: Path | None = None,
    public_factors_csv: Path | None = None,
    cfg: SOPConfig | None = None,
    *,
    long_weight: float = 1.0,
    short_weight: float = 1.0,
    r_gov_override: pd.Series | None = None,
    r_cp_override: pd.Series | None = None,
) -> tuple[pd.DataFrame, list[dict], SOPConfig, pd.Series]:
    """
    日频面板：r_gov/r_cp、离散 Pos、StratRet（含滑点）。
    返回 (daily_df, selection_log_rows, cfg, pos_exe_turn_series 用于统计换仓次数)。

    long_weight / short_weight：持有国债/短融时的收益倍数（名义杠杆）。
    r_gov_override / r_cp_override：可选日收益序列（与 pub 日历对齐），用于净价等替换 CSI 日收益。
    """
    ix = input_xlsx or INPUT_XLSX
    pcsv = public_factors_csv or PUBLIC_FACTORS_CSV
    cfg = cfg or SOPConfig()
    lw = float(long_weight)
    sw = float(short_weight)
    if lw <= 0 or sw <= 0:
        raise ValueError("long_weight and short_weight must be positive")

    if not ix.exists():
        raise FileNotFoundError(ix)
    if not pcsv.exists():
        raise FileNotFoundError(pcsv)

    price_df = pd.read_excel(ix, sheet_name="Price")
    fac_df = pd.read_excel(ix, sheet_name="Factors")
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    fac_df["Date"] = pd.to_datetime(fac_df["Date"])
    price = price_df.sort_values("Date").set_index("Date")["Close"].astype(float)
    factors = fac_df.sort_values("Date").set_index("Date").select_dtypes(include=["number"]).ffill()

    selector = SOPSelector(price=price, factors=factors, cfg=cfg)

    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    start = max(start, price.index.min())
    end = min(end, price.index.max())
    rebal_dates = pd.date_range(start=start, end=end, freq=f"{cfg.rebalance_months}ME")
    if len(rebal_dates) == 0 or start < rebal_dates[0]:
        rebal_dates = pd.DatetimeIndex([pd.Timestamp(start)]).append(rebal_dates)

    signal_segments: list[pd.DataFrame] = []
    log_rows: list[dict] = []
    for i in range(len(rebal_dates) - 1):
        curr = pd.Timestamp(rebal_dates[i])
        nxt = pd.Timestamp(rebal_dates[i + 1])
        train_end = curr
        train_start = train_end - pd.DateOffset(years=cfg.lookback_years)

        selected = selector.screen_factors(train_start, train_end)
        log_rows.append(
            {
                "Date": curr.date().isoformat(),
                "SelectedCount": int(len(selected)),
                "Factors": "|".join(selected.get("Factor", pd.Series([], dtype=str)).astype(str).tolist()),
            }
        )

        test_buffer = curr - pd.Timedelta(days=400)
        f_chunk = selector.df_factors.loc[test_buffer:nxt]
        if selected is not None and not selected.empty:
            sig_df = selector.generate_signals(f_chunk, selected)
        else:
            sig_df = pd.DataFrame(0.0, index=f_chunk.index, columns=["Static", "Dynamic"])

        mask = (sig_df.index >= curr) & (sig_df.index < nxt if i < len(rebal_dates) - 2 else sig_df.index <= nxt)
        signal_segments.append(sig_df.loc[mask])

    full_sig = pd.concat(signal_segments).sort_index()
    full_sig = full_sig[~full_sig.index.duplicated(keep="first")]

    dyn = full_sig["Dynamic"].copy()
    pos = pd.Series(0.0, index=dyn.index)
    pos[dyn > cfg.pos_threshold] = 1.0
    pos[dyn < -cfg.pos_threshold] = -1.0

    pub = pd.read_csv(pcsv)
    pub["Date"] = pd.to_datetime(pub["Date"])
    pub = pub.sort_values("Date")
    for c in (COL_GOV, COL_CP):
        if c not in pub.columns:
            raise KeyError(f"Missing {c} in {pcsv}")
    gov = pd.to_numeric(pub[COL_GOV], errors="coerce")
    cp = pd.to_numeric(pub[COL_CP], errors="coerce")
    rl = gov.pct_change().fillna(0.0)
    rs = cp.pct_change().fillna(0.0)

    px_dates = pub["Date"]
    df = pd.DataFrame({"Date": px_dates, "r_gov": rl.values, "r_cp": rs.values})
    df = df.set_index("Date")
    df = df.loc[df.index.intersection(pos.index)].copy()
    if r_gov_override is not None:
        rg = pd.to_numeric(r_gov_override.reindex(df.index), errors="coerce").ffill().bfill().fillna(0.0)
        df["r_gov"] = rg
    if r_cp_override is not None:
        rc = pd.to_numeric(r_cp_override.reindex(df.index), errors="coerce").ffill().bfill().fillna(0.0)
        df["r_cp"] = rc

    df["Signal_Dynamic"] = dyn.reindex(df.index).fillna(0.0)
    df["Pos"] = pos.reindex(df.index).fillna(0.0)
    pos_exe = df["Pos"].shift(1).fillna(0.0)
    strat_ret = pd.Series(0.0, index=df.index)
    strat_ret[pos_exe > 0] = lw * df.loc[pos_exe > 0, "r_gov"]
    strat_ret[pos_exe < 0] = sw * df.loc[pos_exe < 0, "r_cp"]
    turn = pos_exe.diff().abs().fillna(0.0)
    strat_ret = strat_ret - cfg.slip_per_turn * (turn > 1e-12).astype(float)
    df["StratRet"] = strat_ret
    df["LongOnlyRet"] = lw * df["r_gov"]
    return df, log_rows, cfg, turn


def compute_v2_daily_continuous_panel(
    input_xlsx: Path | None = None,
    public_factors_csv: Path | None = None,
    cfg: SOPConfig | None = None,
    *,
    long_weight: float = 1.0,
    short_weight: float = 1.0,
    r_gov_override: pd.Series | None = None,
    r_cp_override: pd.Series | None = None,
    cont_gamma: float = 1.0,
    w_floor: float = 0.8,
    w_cap: float = 1.2,
    z_win: int = 120,
    z_minp: int = 30,
    hmm_cfg: HMMConfig | None = None,
) -> tuple[pd.DataFrame, list[dict], SOPConfig, pd.Series]:
    """
    V2 连续仓位：Dynamic 信号先做滚动标准化，再经 tanh 映射为 w_gov,w_cp∈[w_floor,w_cap] 且和为 w_floor+w_cap。
    """
    from gov_cp_continuous_weights import (
        bipartite_weights_from_z,
        daily_strategy_return_continuous,
        signal_to_zscore,
    )
    from gov_cp_hmm_overlay import HMMOverlayConfig, compute_hmm_overlay

    ix = input_xlsx or INPUT_XLSX
    pcsv = public_factors_csv or PUBLIC_FACTORS_CSV
    cfg = cfg or SOPConfig()
    lw = float(long_weight)
    sw = float(short_weight)
    if lw <= 0 or sw <= 0:
        raise ValueError("long_weight and short_weight must be positive")

    if not ix.exists():
        raise FileNotFoundError(ix)
    if not pcsv.exists():
        raise FileNotFoundError(pcsv)

    price_df = pd.read_excel(ix, sheet_name="Price")
    fac_df = pd.read_excel(ix, sheet_name="Factors")
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    fac_df["Date"] = pd.to_datetime(fac_df["Date"])
    price = price_df.sort_values("Date").set_index("Date")["Close"].astype(float)
    factors = fac_df.sort_values("Date").set_index("Date").select_dtypes(include=["number"]).ffill()

    selector = SOPSelector(price=price, factors=factors, cfg=cfg)

    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    start = max(start, price.index.min())
    end = min(end, price.index.max())
    rebal_dates = pd.date_range(start=start, end=end, freq=f"{cfg.rebalance_months}ME")
    if len(rebal_dates) == 0 or start < rebal_dates[0]:
        rebal_dates = pd.DatetimeIndex([pd.Timestamp(start)]).append(rebal_dates)

    signal_segments: list[pd.DataFrame] = []
    log_rows: list[dict] = []
    for i in range(len(rebal_dates) - 1):
        curr = pd.Timestamp(rebal_dates[i])
        nxt = pd.Timestamp(rebal_dates[i + 1])
        train_end = curr
        train_start = train_end - pd.DateOffset(years=cfg.lookback_years)

        selected = selector.screen_factors(train_start, train_end)
        log_rows.append(
            {
                "Date": curr.date().isoformat(),
                "SelectedCount": int(len(selected)),
                "Factors": "|".join(selected.get("Factor", pd.Series([], dtype=str)).astype(str).tolist()),
            }
        )

        test_buffer = curr - pd.Timedelta(days=400)
        f_chunk = selector.df_factors.loc[test_buffer:nxt]
        if selected is not None and not selected.empty:
            sig_df = selector.generate_signals(f_chunk, selected)
        else:
            sig_df = pd.DataFrame(0.0, index=f_chunk.index, columns=["Static", "Dynamic"])

        mask = (sig_df.index >= curr) & (sig_df.index < nxt if i < len(rebal_dates) - 2 else sig_df.index <= nxt)
        signal_segments.append(sig_df.loc[mask])

    full_sig = pd.concat(signal_segments).sort_index()
    full_sig = full_sig[~full_sig.index.duplicated(keep="first")]
    dyn = full_sig["Dynamic"].copy()

    pub = pd.read_csv(pcsv)
    pub["Date"] = pd.to_datetime(pub["Date"])
    pub = pub.sort_values("Date")
    for c in (COL_GOV, COL_CP):
        if c not in pub.columns:
            raise KeyError(f"Missing {c} in {pcsv}")
    gov = pd.to_numeric(pub[COL_GOV], errors="coerce")
    cp = pd.to_numeric(pub[COL_CP], errors="coerce")
    rl = gov.pct_change().fillna(0.0)
    rs = cp.pct_change().fillna(0.0)

    px_dates = pub["Date"]
    df = pd.DataFrame({"Date": px_dates, "r_gov": rl.values, "r_cp": rs.values})
    df = df.set_index("Date")
    df = df.loc[df.index.intersection(dyn.index)].copy()
    if r_gov_override is not None:
        df["r_gov"] = pd.to_numeric(r_gov_override.reindex(df.index), errors="coerce").ffill().bfill().fillna(0.0)
    if r_cp_override is not None:
        df["r_cp"] = pd.to_numeric(r_cp_override.reindex(df.index), errors="coerce").ffill().bfill().fillna(0.0)

    dyn_i = dyn.reindex(df.index).fillna(0.0)
    df["Signal_Dynamic"] = dyn_i
    z_base = signal_to_zscore(dyn_i, win=z_win, minp=z_minp)
    df["Signal_Z_Base"] = z_base

    hmm_cfg = hmm_cfg or HMMConfig()
    if hmm_cfg.enabled:
        overlay = compute_hmm_overlay(
            z_base,
            df["r_gov"],
            df["r_cp"],
            HMMOverlayConfig(
                n_states=hmm_cfg.n_states,
                fit_window=hmm_cfg.fit_window,
                min_history=hmm_cfg.min_history,
                refit_freq=hmm_cfg.refit_freq,
                vol_window=hmm_cfg.vol_window,
                crowd_window=hmm_cfg.crowd_window,
                shock_window=hmm_cfg.shock_window,
                bias_strength=hmm_cfg.bias_strength,
                scale_strength=hmm_cfg.scale_strength,
                min_scale=hmm_cfg.min_scale,
                max_scale=hmm_cfg.max_scale,
                covariance_type=hmm_cfg.covariance_type,
                n_iter=hmm_cfg.n_iter,
                random_state=hmm_cfg.random_state,
            ),
        )
        z_used = overlay["Signal_Z_HMM"].reindex(df.index).fillna(z_base)
        df["Signal_Z_HMM"] = z_used
        df["HMM_State"] = overlay["HMM_State"].reindex(df.index).astype(float)
        df["HMM_Score"] = overlay["HMM_Score"].reindex(df.index).fillna(0.0)
        df["HMM_Scale"] = overlay["HMM_Scale"].reindex(df.index).fillna(1.0)
        df["HMM_Bias"] = overlay["HMM_Bias"].reindex(df.index).fillna(0.0)
    else:
        z_used = z_base
        df["Signal_Z_HMM"] = z_base
        df["HMM_State"] = np.nan
        df["HMM_Score"] = 0.0
        df["HMM_Scale"] = 1.0
        df["HMM_Bias"] = 0.0

    w_gov, w_cp = bipartite_weights_from_z(z_used, gamma=cont_gamma, w_floor=w_floor, w_cap=w_cap)
    w_gov = w_gov * lw
    w_cp = w_cp * sw
    strat_ret, wg_e, wc_e = daily_strategy_return_continuous(
        df["r_gov"], df["r_cp"], w_gov, w_cp, slip_per_unit_turnover=cfg.slip_per_turn
    )
    df["w_gov"] = w_gov
    df["w_cp"] = w_cp
    df["w_gov_exe"] = wg_e
    df["w_cp_exe"] = wc_e
    df["StratRet"] = strat_ret
    df["LongOnlyRet"] = lw * df["r_gov"]
    net = w_gov - w_cp
    df["Pos"] = np.sign(net).replace(0.0, np.nan).fillna(0.0)
    turn = w_gov.diff().abs().fillna(0.0) + w_cp.diff().abs().fillna(0.0)
    return df, log_rows, cfg, turn


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    daily_df, log_rows, cfg, turn_daily = compute_v2_daily_strategy_panel()
    pd.DataFrame(log_rows).to_csv(OUT_SEL_LOG, index=False, encoding="utf-8-sig")

    tw_ls = timing_win_rates_from_position(daily_df["r_gov"], daily_df["r_cp"], daily_df["Pos"])
    tw_lo = longonly_buyhold_win_rate(daily_df["r_gov"])

    df = daily_df
    # 周频压缩后评估
    wk = pd.DataFrame(index=df.index)
    wk["LongOnlyRet"] = (1.0 + df["LongOnlyRet"]).resample("W-FRI").prod() - 1.0
    wk["StratRet"] = (1.0 + df["StratRet"]).resample("W-FRI").prod() - 1.0
    wk = wk.dropna(how="all")
    df = wk.copy()
    df["LongOnlyNav"] = (1.0 + df["LongOnlyRet"]).cumprod()
    df["LongShortNav"] = (1.0 + df["StratRet"]).cumprod()

    years = sample_years(df.reset_index()["Date"])
    metrics = {
        "样本起止": f"{df.index.min().date()} ~ {df.index.max().date()}",
        "样本跨度（年）": f"{years:.3f}",
        "LongOnly 近似年化": f"{cagr_from_nav(df['LongOnlyNav'], years)*100:.3f}%",
        "LongShort 近似年化": f"{cagr_from_nav(df['LongShortNav'], years)*100:.3f}%",
        "LongOnly 年化波动": f"{ann_vol_daily(df['LongOnlyRet'])*100:.3f}%",
        "LongShort 年化波动": f"{ann_vol_daily(df['StratRet'])*100:.3f}%",
        "LongOnly 最大回撤": f"{max_drawdown(df['LongOnlyNav'])*100:.2f}%",
        "LongShort 最大回撤": f"{max_drawdown(df['LongShortNav'])*100:.2f}%",
        "换仓次数（估计）": f"{int((turn_daily > 1e-12).sum())}",
        "滑点（每次）": f"{cfg.slip_per_turn*100:.2f}%",
        "择时（日频）：LongOnly 多头日胜率": fmt_pct(tw_lo["timing_long_win_rate"]),
        "择时（日频）：LongShort 持国债日胜率": fmt_pct(tw_ls["timing_long_win_rate"]),
        "择时（日频）：LongShort 持短融日胜率": fmt_pct(tw_ls["timing_short_win_rate"]),
    }

    # plots + outputs
    plot_strat(df.reset_index(), df["LongOnlyNav"], df["LongShortNav"], {"cagr_lo": cagr_from_nav(df["LongOnlyNav"], years), "cagr_ls": cagr_from_nav(df["LongShortNav"], years)})
    pd.DataFrame({"Date": df.index, "LongOnly_nav": df["LongOnlyNav"].values, "LongShort_nav": df["LongShortNav"].values}).to_csv(
        OUT_STRAT_CSV, index=False, encoding="utf-8-sig"
    )

    strat_rows = [
        [
            "LongOnly",
            "买入并持有国债活跃券指数（931018）",
            metrics["LongOnly 近似年化"],
            metrics["LongOnly 年化波动"],
            metrics["LongOnly 最大回撤"],
            fmt_pct(tw_lo["timing_long_win_rate"]),
            "—",
        ],
        [
            "LongShort",
            "收益率筛选信号驱动：多国债/多短融二选一切换",
            metrics["LongShort 近似年化"],
            metrics["LongShort 年化波动"],
            metrics["LongShort 最大回撤"],
            fmt_pct(tw_ls["timing_long_win_rate"]),
            fmt_pct(tw_ls["timing_short_win_rate"]),
        ],
    ]
    strat_table_html = f"""
  <p class="muted">信号：收益率筛选（Dynamic）→ 离散仓位阈值 ±{cfg.pos_threshold}；换仓次数≈{metrics['换仓次数（估计）']}；滑点 {metrics['滑点（每次）']}。
  择时胜率说明（日频、不含票息拆分）：持国债日统计国债日收益&gt;0 占比；持短融日统计短融日收益&gt;0 占比。</p>
  <table class="stats">
    <tr><th>策略</th><th>口径</th><th>近似年化</th><th>年化波动</th><th>最大回撤</th><th>多头胜率</th><th>空头胜率</th></tr>
    <tr><td>{strat_rows[0][0]}</td><td>{strat_rows[0][1]}</td><td style="text-align:right">{strat_rows[0][2]}</td><td style="text-align:right">{strat_rows[0][3]}</td><td style="text-align:right">{strat_rows[0][4]}</td><td style="text-align:right">{strat_rows[0][5]}</td><td style="text-align:right">{strat_rows[0][6]}</td></tr>
    <tr><td>{strat_rows[1][0]}</td><td>{strat_rows[1][1]}</td><td style="text-align:right">{strat_rows[1][2]}</td><td style="text-align:right">{strat_rows[1][3]}</td><td style="text-align:right">{strat_rows[1][4]}</td><td style="text-align:right">{strat_rows[1][5]}</td><td style="text-align:right">{strat_rows[1][6]}</td></tr>
  </table>
"""

    html_doc = build_report_html(metrics, strat_table_html)
    OUT_HTML.write_text(html_doc, encoding="utf-8")

    html_metrics_rows = [[k, v] for k, v in metrics.items()]
    build_pdf(html_metrics_rows, strat_rows)

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

