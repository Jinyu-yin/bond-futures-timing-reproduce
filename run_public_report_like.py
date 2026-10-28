from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INPUT = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_DIR = ROOT / "reproduce" / "report_like"
OUT_DIR_FIG = ROOT / "reproduce" / "figures"


@dataclass(frozen=True)
class ReportLikeConfig:
    input_xlsx: Path = PUBLIC_INPUT
    start: str = "2016-01-01"

    # report fixed defaults
    signal_mode: str = "or"  # report: change>0 OR strength>0
    reverse_signal: bool = False  # report: do NOT reverse; disable instead

    # multi-period momentum (trading days): 2w and 12m
    mom_short: int = 10
    mom_long: int = 252

    # topN per group
    top_1: int = 15
    top_2: int = 15
    top_3: int = 15
    top_4: int = 5

    # threshold on 4-group sum
    open_threshold: int = 0  # also test 1

    # signal construction
    z_win_level: int = 252
    z_minp_level: int = 60
    z_win_diff: int = 63
    z_minp_diff: int = 20

    # friction: slippage
    # report text: "买卖各1滑点". We implement as 0.001 per side.
    # double-sided per full turn cost = 2 * slippage_per_side * |Δpos|
    slippage_per_side: float = 0.001

    # lag proxy (day-level) for report minute lag robustness
    trade_lag_days: int = 1  # base uses shift(1)

    # factor disable (instead of reverse)
    disable_lookback_days: int = 252
    disable_threshold_nav_ret: float = 0.0  # if 12m factor nav return <0 => disable


def ts_zscore(s: pd.Series, win: int, minp: int) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std().replace(0, np.nan)
    return (s - mu) / sd


def build_single_factor_signal(x: pd.Series, cfg: ReportLikeConfig) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    strength = ts_zscore(x, win=cfg.z_win_level, minp=cfg.z_minp_level)
    change = ts_zscore(x.diff(1), win=cfg.z_win_diff, minp=cfg.z_minp_diff)

    if cfg.signal_mode == "and":
        cond = (change > 0) & (strength > 0)
    else:
        cond = (change > 0) | (strength > 0)

    # report example: cond => short (-1), else long (+1)
    sig = pd.Series(np.where(cond, -1.0, 1.0), index=x.index, dtype=float)
    sig = sig.where(x.notna(), 0.0)
    return sig.fillna(0.0)


def month_first_trading_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=idx)
    first = s.groupby(s.index.to_period("M")).apply(lambda x: x.index.min())
    return pd.DatetimeIndex(first.values)


def next_trading_day(dates: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp:
    pos = dates.searchsorted(d, side="left")
    if pos >= len(dates):
        return dates[-1]
    if dates[pos] == d:
        if pos + 1 < len(dates):
            return dates[pos + 1]
        return dates[-1]
    return dates[pos]


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


def apply_slippage_cost(ret: pd.Series, pos: pd.Series, cfg: ReportLikeConfig) -> pd.Series:
    # report: 0.001 slippage per side; charge on position change
    p = pos.fillna(0.0)
    turn = p.diff().abs().fillna(0.0)
    return ret - (2.0 * cfg.slippage_per_side) * turn


def prepare_data(cfg: ReportLikeConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], dict[str, list[str]]]:
    price = pd.read_excel(cfg.input_xlsx, sheet_name="Price")
    factors = pd.read_excel(cfg.input_xlsx, sheet_name="Factors")

    price["Date"] = pd.to_datetime(price["Date"])
    factors["Date"] = pd.to_datetime(factors["Date"])

    px = price[["Date", "Close"]].dropna().sort_values("Date").set_index("Date")
    px = px.loc[px.index >= pd.Timestamp(cfg.start)]
    ret = px["Close"].pct_change().fillna(0.0)

    f = factors.sort_values("Date").set_index("Date").select_dtypes(include=["number"])
    f = f.reindex(px.index).ffill(limit=30)

    cols = [c for c in f.columns if str(c).strip()]
    gcols = {
        "1_growth": [c for c in cols if str(c).startswith("1_")],
        "2_inflation": [c for c in cols if str(c).startswith("2_")],
        "3_credit": [c for c in cols if str(c).startswith("3_")],
        "4_omo": [c for c in cols if str(c).startswith("4_")],
    }
    return px, ret, f, cols, gcols


def build_sig_and_factor_nav(px: pd.DataFrame, ret: pd.Series, f: pd.DataFrame, cols: list[str], cfg: ReportLikeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig = pd.concat({c: build_single_factor_signal(f[c], cfg) for c in cols}, axis=1).fillna(0.0)
    sig.columns = cols
    pnl = sig.shift(cfg.trade_lag_days).mul(ret, axis=0).fillna(0.0)
    nav_factor = (1.0 + pnl).cumprod()
    return sig, nav_factor


def factor_enabled_mask(nav_factor: pd.DataFrame, asof_idx: int, cfg: ReportLikeConfig) -> pd.Series:
    # Disable if 12m momentum of factor nav is negative (proxy for report's "don't reverse, don't use")
    win = cfg.disable_lookback_days
    if asof_idx - win < 0:
        return pd.Series(True, index=nav_factor.columns)
    end = asof_idx
    start = asof_idx - win
    mom = nav_factor.iloc[end] / nav_factor.iloc[start] - 1.0
    return (mom >= cfg.disable_threshold_nav_ret).fillna(False)


def build_rebalance_calendar(px_dates: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    # Report text:
    # - OMO (daily updated) => monthly first trading day
    # - others => next trading day after signal update (we proxy with factor AvailDate logic elsewhere)
    # For speed and public data limitations, we will:
    # - keep OMO monthly
    # - keep others monthly for now (will be refined by AvailDate in next step)
    return {
        "4_omo": month_first_trading_days(px_dates),
        "1_growth": month_first_trading_days(px_dates),
        "2_inflation": month_first_trading_days(px_dates),
        "3_credit": month_first_trading_days(px_dates),
    }


def run_report_like(cfg: ReportLikeConfig) -> pd.DataFrame:
    px, ret, f, cols, gcols = prepare_data(cfg)
    sig, nav_factor = build_sig_and_factor_nav(px, ret, f, cols, cfg)

    topn = {
        "1_growth": cfg.top_1,
        "2_inflation": cfg.top_2,
        "3_credit": cfg.top_3,
        "4_omo": cfg.top_4,
    }

    rebals = build_rebalance_calendar(px.index)
    group_sig = pd.DataFrame(0.0, index=px.index, columns=list(gcols.keys()))
    dates = px.index

    # Evaluate each group on its own rebal dates
    for g, gc in gcols.items():
        reb = rebals[g]
        for i, d in enumerate(reb):
            d_next = reb[i + 1] if i + 1 < len(reb) else dates[-1] + pd.Timedelta(days=1)
            mask = (dates >= d) & (dates < d_next)
            at = int(dates.get_loc(d))
            last = at - 1
            if last < max(cfg.mom_long, cfg.mom_short, 5):
                group_sig.loc[mask, g] = 0.0
                continue

            if not gc:
                group_sig.loc[mask, g] = 0.0
                continue

            mom_s = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_short, last) for c in gc})
            mom_l = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_long, last) for c in gc})
            cr = composite_rank(mom_s, mom_l).dropna()
            if cr.empty:
                group_sig.loc[mask, g] = 0.0
                continue

            enabled = factor_enabled_mask(nav_factor[gc], last, cfg)
            cr = cr[enabled.reindex(cr.index).fillna(False)]
            if cr.empty:
                group_sig.loc[mask, g] = 0.0
                continue

            n = min(topn[g], len(cr))
            chosen = cr.sort_values().head(n).index.tolist()
            # report: remove negative 12m momentum factors from pool
            chosen = [c for c in chosen if pd.notna(mom_l.get(c)) and float(mom_l[c]) >= 0.0]
            if not chosen:
                group_sig.loc[mask, g] = 0.0
                continue

            ssum = sig.loc[mask, chosen].sum(axis=1)
            gs = pd.Series(0.0, index=dates[mask])
            gs[ssum > 0] = 1.0
            gs[ssum < 0] = -1.0
            group_sig.loc[mask, g] = gs

    gsum = group_sig.sum(axis=1)
    final_pos = pd.Series(0.0, index=px.index)
    if cfg.open_threshold <= 0:
        final_pos[gsum > 0] = 1.0
        final_pos[gsum < 0] = -1.0
    else:
        final_pos[gsum > cfg.open_threshold] = 1.0
        final_pos[gsum < -cfg.open_threshold] = -1.0

    strat_ret = final_pos.shift(1).fillna(0.0) * ret
    strat_ret = apply_slippage_cost(strat_ret, final_pos, cfg)
    strat_nav = (1.0 + strat_ret).cumprod()

    group_ret = group_sig.shift(1).fillna(0.0).mul(ret, axis=0)
    group_nav = (1.0 + group_ret).cumprod()

    out = pd.DataFrame(index=px.index)
    out["Close"] = px["Close"]
    out["Ret"] = ret
    out["BenchNav"] = (1.0 + ret).cumprod()
    out["GroupSum"] = gsum
    out["FinalPos"] = final_pos
    out["StrategyRet"] = strat_ret
    out["StrategyNav"] = strat_nav
    for g in group_sig.columns:
        out[f"{g}_sig"] = group_sig[g]
        out[f"{g}_nav"] = group_nav[g]

    out.reset_index(names="Date", inplace=True)
    return out


def calc_metrics(ret: pd.Series) -> dict[str, float]:
    r = ret.dropna()
    if len(r) == 0:
        return {"ann": np.nan, "vol": np.nan, "sharpe": np.nan, "mdd": np.nan, "calmar": np.nan}
    ann = (1 + r.mean()) ** 252 - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 1e-12 else np.nan
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    mdd = float(dd.min())
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return {
        "ann": float(ann),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(calmar),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # threshold 0 and 1
    metrics_rows: list[dict[str, object]] = []
    best = None

    for th in [0, 1]:
        df = run_report_like(ReportLikeConfig(open_threshold=th))
        out_csv = OUT_DIR / f"nav_report_like_th{th}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        m = calc_metrics(df["StrategyRet"])
        metrics_rows.append({"variant": f"th{th}", **m, "nonzero_ratio": float((df["FinalPos"] != 0).mean())})
        best = df if best is None else best
        print("saved", out_csv)

    pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "metrics_report_like.csv", index=False, encoding="utf-8-sig")
    print("saved", OUT_DIR / "metrics_report_like.csv")


if __name__ == "__main__":
    main()
