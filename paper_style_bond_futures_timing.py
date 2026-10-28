from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    input_xlsx: Path
    out_csv: Path

    start: str = "2016-01-01"

    # multi-period momentum (trading days)
    mom_short: int = 10  # ~2 weeks
    mom_long: int = 252  # ~12 months

    # topN per group (report text)
    top_1: int = 15
    top_2: int = 15
    top_3: int = 15
    top_4: int = 5

    # composite threshold on sum of 4 group signals in {-4..4}
    # 0: trade when sum != 0 (long if >0, short if <0) — 相对「更敢出手」
    # 1: long only if sum>1, short only if sum<-1 — 更保守
    open_threshold: int = 0

    # friction: each unit change in **effective** position costs cost_per_turn * |Δeff_pos|
    cost_per_turn: float = 0.00005

    # signal construction (trend strength / trend change)
    z_win_level: int = 252
    z_minp_level: int = 60
    z_win_diff: int = 63
    z_minp_diff: int = 20

    # Price column on sheet "Price": default 国债期货收盘；CloseNet=中债净价指数对齐到交易日（近似不含票息）
    price_close_col: str = "Close"

    # 名义杠杆：信号仍为 ±1，回测用 eff_pos=±weight 乘日收益；换手成本按 |Δeff_pos| 计
    long_weight: float = 1.0
    short_weight: float = 1.0


def ts_zscore(s: pd.Series, win: int, minp: int) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std().replace(0, np.nan)
    return (s - mu) / sd


def build_single_factor_signal(x: pd.Series, cfg: Config) -> pd.Series:
    """
    Report example (PMI):
    if (trend_change > 0) OR (trend_strength > 0) => short (-1), else long (+1).

    Proxy:
    - trend_change: zscore of 1-day diff
    - trend_strength: zscore of level
    """
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


def apply_cost(ret: pd.Series, pos: pd.Series, cost_per_turn: float) -> pd.Series:
    p = pos.fillna(0.0)
    turn = p.diff().abs().fillna(0.0)
    return ret - cost_per_turn * turn


def directional_win_rates(ret: pd.Series, final_pos: pd.Series) -> dict[str, float]:
    """
    按「交易日持有方向」拆分胜率：多头日看标的涨跌幅是否 >0，空头日是否 <0。
    使用与策略一致的押后一日仓位 final_pos.shift(1)。
    """
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    pos = pd.to_numeric(final_pos, errors="coerce").fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    long_m = prev > 0
    short_m = prev < 0
    inv_m = prev != 0

    long_wr = float((r[long_m] > 0).mean()) if bool(long_m.any()) else float("nan")
    short_wr = float((r[short_m] < 0).mean()) if bool(short_m.any()) else float("nan")
    hit_inv = ((prev > 0) & (r > 0)) | ((prev < 0) & (r < 0))
    inv_wr = float(hit_inv[inv_m].mean()) if bool(inv_m.any()) else float("nan")

    return {
        "n_long_days": float(long_m.sum()),
        "n_short_days": float(short_m.sum()),
        "n_invested_days": float(inv_m.sum()),
        "long_win_rate": long_wr,
        "short_win_rate": short_wr,
        "invested_win_rate": inv_wr,
    }


def run(cfg: Config) -> pd.DataFrame:
    if not cfg.input_xlsx.exists():
        raise FileNotFoundError(cfg.input_xlsx)

    price = pd.read_excel(cfg.input_xlsx, sheet_name="Price")
    factors = pd.read_excel(cfg.input_xlsx, sheet_name="Factors")

    price["Date"] = pd.to_datetime(price["Date"])
    factors["Date"] = pd.to_datetime(factors["Date"])

    close_col = cfg.price_close_col
    if close_col not in price.columns:
        raise ValueError(
            f"Price sheet has no column {close_col!r}; "
            f"build public_input_data.xlsx with CloseNet or use Close. Columns: {list(price.columns)}"
        )

    px = price[["Date", close_col]].dropna().sort_values("Date").rename(columns={close_col: "_close"})
    px = px.set_index("Date")
    px = px.loc[px.index >= pd.Timestamp(cfg.start)]
    ret = px["_close"].pct_change().fillna(0.0)

    f = factors.sort_values("Date").set_index("Date").select_dtypes(include=["number"])
    f = f.reindex(px.index).ffill(limit=30)

    cols = [c for c in f.columns if str(c).strip()]
    gcols = {
        "1_growth": [c for c in cols if str(c).startswith("1_")],
        "2_inflation": [c for c in cols if str(c).startswith("2_")],
        "3_credit": [c for c in cols if str(c).startswith("3_")],
        "4_omo": [c for c in cols if str(c).startswith("4_")],
    }
    topn = {
        "1_growth": cfg.top_1,
        "2_inflation": cfg.top_2,
        "3_credit": cfg.top_3,
        "4_omo": cfg.top_4,
    }

    sig = pd.concat({c: build_single_factor_signal(f[c], cfg) for c in cols}, axis=1)
    sig.columns = cols
    sig = sig.fillna(0.0)

    pnl = sig.shift(1).mul(ret, axis=0).fillna(0.0)
    nav_factor = (1.0 + pnl).cumprod()

    reb = month_first_trading_days(px.index)
    group_sig = pd.DataFrame(0.0, index=px.index, columns=list(gcols.keys()))

    dates = px.index
    for i, d in enumerate(reb):
        d_next = reb[i + 1] if i + 1 < len(reb) else dates[-1] + pd.Timedelta(days=1)
        mask = (dates >= d) & (dates < d_next)

        at = int(dates.get_loc(d))
        last = at - 1
        if last < max(cfg.mom_long, cfg.mom_short, 5):
            group_sig.loc[mask, :] = 0.0
            continue

        for g, gc in gcols.items():
            if not gc:
                group_sig.loc[mask, g] = 0.0
                continue

            mom_s = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_short, last) for c in gc})
            mom_l = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_long, last) for c in gc})
            cr = composite_rank(mom_s, mom_l).dropna()
            if cr.empty:
                group_sig.loc[mask, g] = 0.0
                continue

            n = min(topn[g], len(cr))
            chosen = cr.sort_values().head(n).index.tolist()
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

    lw = float(cfg.long_weight)
    sw = float(cfg.short_weight)
    if lw <= 0 or sw <= 0:
        raise ValueError("long_weight and short_weight must be positive")

    eff_pos = final_pos.replace({1.0: lw, -1.0: -sw, 0.0: 0.0})

    strat_ret = eff_pos.shift(1).fillna(0.0) * ret
    strat_ret = apply_cost(strat_ret, eff_pos, cfg.cost_per_turn)
    strat_nav = (1.0 + strat_ret).cumprod()

    group_ret = group_sig.shift(1).fillna(0.0).mul(ret, axis=0)
    group_nav = (1.0 + group_ret).cumprod()

    out = pd.DataFrame(index=px.index)
    out["Close"] = px["_close"]
    out["Ret"] = ret
    out["BenchNav"] = (1.0 + ret).cumprod()
    out["GroupSum"] = gsum
    out["FinalPos"] = final_pos
    if lw != 1.0 or sw != 1.0:
        out["EffPos"] = eff_pos
    out["StrategyRet"] = strat_ret
    out["StrategyNav"] = strat_nav
    for g in group_sig.columns:
        out[f"{g}_sig"] = group_sig[g]
        out[f"{g}_nav"] = group_nav[g]

    out.reset_index(names="Date", inplace=True)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = Config(
        input_xlsx=root / "input_data.xlsx",
        out_csv=root / "reproduce" / "paper_style_nav.csv",
        open_threshold=0,
        cost_per_turn=0.00005,
    )
    df = run(cfg)
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.out_csv, index=False, encoding="utf-8-sig")
    wr = directional_win_rates(df["Ret"], df["FinalPos"])
    print("saved", cfg.out_csv, "rows", len(df))
    print(
        "win_rates",
        {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v) for k, v in wr.items()},
    )


if __name__ == "__main__":
    main()
