from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INPUT = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_RESULTS = ROOT / "reproduce" / "public_optimization_results.csv"
OUT_BEST = ROOT / "reproduce" / "paper_style_nav_public_best.csv"


@dataclass(frozen=True)
class SweepConfig:
    input_xlsx: Path
    start: str = "2016-01-01"
    mom_short: int = 10
    mom_long: int = 252
    top_1: int = 15
    top_2: int = 15
    top_3: int = 15
    top_4: int = 5
    open_threshold: int = 0
    cost_per_turn: float = 0.00005
    z_win_level: int = 252
    z_minp_level: int = 60
    z_win_diff: int = 63
    z_minp_diff: int = 20
    signal_mode: str = "or"
    reverse_signal: bool = False


@dataclass(frozen=True)
class PreparedData:
    px: pd.DataFrame
    ret: pd.Series
    factors: pd.DataFrame
    cols: list[str]
    gcols: dict[str, list[str]]


def ts_zscore(s: pd.Series, win: int, minp: int) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std().replace(0, np.nan)
    return (s - mu) / sd


def build_single_factor_signal(x: pd.Series, cfg: SweepConfig) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    strength = ts_zscore(x, win=cfg.z_win_level, minp=cfg.z_minp_level)
    change = ts_zscore(x.diff(1), win=cfg.z_win_diff, minp=cfg.z_minp_diff)

    if cfg.signal_mode == "and":
        cond = (change > 0) & (strength > 0)
    else:
        cond = (change > 0) | (strength > 0)

    sig = pd.Series(np.where(cond, -1.0, 1.0), index=x.index, dtype=float)
    if cfg.reverse_signal:
        sig = -sig
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


def score_result(metrics: dict[str, float]) -> float:
    sharpe = metrics.get("sharpe", np.nan)
    ann = metrics.get("ann", np.nan)
    mdd = abs(metrics.get("mdd", np.nan))
    if np.isnan(sharpe) or np.isnan(ann) or np.isnan(mdd):
        return -np.inf
    return sharpe + 0.3 * ann - 0.2 * mdd


def prepare_data(cfg: SweepConfig) -> PreparedData:
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
    return PreparedData(px=px, ret=ret, factors=f, cols=cols, gcols=gcols)


def run_one(cfg: SweepConfig, data: PreparedData) -> pd.DataFrame:
    px = data.px
    ret = data.ret
    f = data.factors
    cols = data.cols
    gcols = data.gcols

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
                continue
            mom_s = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_short, last) for c in gc})
            mom_l = pd.Series({c: momentum_return(nav_factor[c], cfg.mom_long, last) for c in gc})
            cr = composite_rank(mom_s, mom_l).dropna()
            if cr.empty:
                continue

            n = min(topn[g], len(cr))
            chosen = cr.sort_values().head(n).index.tolist()
            chosen = [c for c in chosen if pd.notna(mom_l.get(c)) and float(mom_l[c]) >= 0.0]
            if not chosen:
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
    strat_ret = apply_cost(strat_ret, final_pos, cfg.cost_per_turn)
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


def build_grid() -> list[SweepConfig]:
    grid = []
    for mom_short, mom_long, threshold, signal_mode, reverse_signal, top_1, top_2, top_3, top_4 in product(
        [5, 10, 20],
        [126, 252],
        [0, 1],
        ["or", "and"],
        [False, True],
        [5, 10, 15],
        [5, 10, 15],
        [3, 5, 10],
        [3, 5],
    ):
        if mom_short >= mom_long:
            continue
        grid.append(
            SweepConfig(
                input_xlsx=PUBLIC_INPUT,
                mom_short=mom_short,
                mom_long=mom_long,
                open_threshold=threshold,
                signal_mode=signal_mode,
                reverse_signal=reverse_signal,
                top_1=top_1,
                top_2=top_2,
                top_3=top_3,
                top_4=top_4,
            )
        )
    return grid


def main() -> None:
    if not PUBLIC_INPUT.exists():
        raise FileNotFoundError(f"Missing {PUBLIC_INPUT}. Run build_public_macro_dataset.py first.")

    base_cfg = SweepConfig(input_xlsx=PUBLIC_INPUT)
    data = prepare_data(base_cfg)

    rows: list[dict[str, float | int | str | bool]] = []
    best_score = -np.inf
    best_df: pd.DataFrame | None = None
    best_cfg: SweepConfig | None = None

    grid = build_grid()
    print(f"total parameter sets: {len(grid)}")
    for i, cfg in enumerate(grid, start=1):
        df = run_one(cfg, data)
        metrics = calc_metrics(df["StrategyRet"])
        nonzero_ratio = float((df["FinalPos"] != 0).mean())
        score = score_result(metrics)
        rows.append(
            {
                "mom_short": cfg.mom_short,
                "mom_long": cfg.mom_long,
                "threshold": cfg.open_threshold,
                "signal_mode": cfg.signal_mode,
                "reverse_signal": cfg.reverse_signal,
                "top_1": cfg.top_1,
                "top_2": cfg.top_2,
                "top_3": cfg.top_3,
                "top_4": cfg.top_4,
                "ann": metrics["ann"],
                "vol": metrics["vol"],
                "sharpe": metrics["sharpe"],
                "mdd": metrics["mdd"],
                "calmar": metrics["calmar"],
                "nonzero_ratio": nonzero_ratio,
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_df = df.copy()
            best_cfg = cfg
        if i % 50 == 0:
            print(f"processed {i} parameter sets; current best score={best_score:.4f}")

    result_df = pd.DataFrame(rows).sort_values(["score", "sharpe", "ann"], ascending=False)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_RESULTS, index=False, encoding="utf-8-sig")

    if best_df is None or best_cfg is None:
        raise RuntimeError("No optimization result generated")
    best_df.to_csv(OUT_BEST, index=False, encoding="utf-8-sig")

    print("saved", OUT_RESULTS)
    print("saved", OUT_BEST)
    print("best parameters:")
    print(best_cfg)
    print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
