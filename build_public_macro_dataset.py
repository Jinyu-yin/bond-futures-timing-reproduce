from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_XLSX = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_FACTORS_CSV = ROOT / "reproduce" / "public_factors.csv"
OUT_PRICE_CSV = ROOT / "reproduce" / "public_price.csv"
START_DATE = "2015-01-01"


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    group_prefix: str
    source: str
    value_col: str
    date_col: str | None = None
    availability: str = "same_day"


def _col_with_keywords(columns: list[str], keywords: list[str]) -> str:
    for col in columns:
        text = str(col)
        if all(k in text for k in keywords):
            return col
    raise KeyError(f"No column matched keywords={keywords}; columns={columns}")


def _to_timestamp(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT

    normalized = (
        s.replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .replace("/", "-")
        .replace(".", "-")
        .replace("\u3000", "")
    )
    normalized = normalized.strip("-")

    if "-" in normalized:
        parts = [p for p in normalized.split("-") if p]
        if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 1
            return pd.Timestamp(year=year, month=month, day=day)

    import re

    digits = re.findall(r"\d+", normalized)
    if len(digits) >= 2:
        year = int(digits[0])
        month = int(digits[1])
        day = int(digits[2]) if len(digits) >= 3 else 1
        return pd.Timestamp(year=year, month=month, day=day)

    return pd.to_datetime(s)


def _availability_date(raw_date: pd.Series, rule: str) -> pd.Series:
    dt = pd.to_datetime(raw_date)
    if rule == "same_day":
        return dt
    if rule == "pmi_month_end":
        return dt + pd.offsets.MonthEnd(0)
    if rule == "next_month_10d":
        return dt + pd.offsets.MonthEnd(1) + pd.Timedelta(days=10)
    if rule == "next_month_12d":
        return dt + pd.offsets.MonthEnd(1) + pd.Timedelta(days=12)
    if rule == "next_month_15d":
        return dt + pd.offsets.MonthEnd(1) + pd.Timedelta(days=15)
    raise ValueError(f"Unknown availability rule: {rule}")


def fetch_price() -> pd.DataFrame:
    import akshare as ak

    df = ak.futures_zh_daily_sina(symbol="T0")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.rename(columns={"date": "Date", "close": "Close"})
    out = out[["Date", "Close"]].dropna().sort_values("Date")
    out = out.loc[out["Date"] >= pd.Timestamp(START_DATE)].reset_index(drop=True)
    return out


def fetch_cbond_net_price_710y_levels() -> pd.Series:
    """
    中债综合指数 — 净价 — 待偿期 7–10 年（日频点位）。
    对齐到期货交易日后可作为 CloseNet，用于近似「不含票息」的资本利得回测口径。
    """
    import akshare as ak

    df = ak.bond_composite_index_cbond(indicator="净价", period="7-10年")
    dt = pd.to_datetime(df["date"])
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=dt).sort_index()
    return s


def attach_close_net_to_price(price: pd.DataFrame) -> pd.DataFrame:
    out = price.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    try:
        net = fetch_cbond_net_price_710y_levels()
        aligned = net.reindex(pd.DatetimeIndex(out["Date"])).ffill()
        out["CloseNet"] = aligned.values
        ok = int(pd.Series(out["CloseNet"]).notna().sum())
        print(f"ok   CloseNet                             (中债净价7-10年 aligned) rows={ok}")
    except Exception as e:
        print(f"warn CloseNet skipped: {e}")
    return out


def fetch_dataset(source: str) -> pd.DataFrame:
    import akshare as ak

    return getattr(ak, source)()


def resolve_date_col(spec: SeriesSpec, columns: list[str]) -> str:
    if spec.date_col is not None:
        return _col_with_keywords(columns, spec.date_col.split("|"))
    if spec.source in {
        "macro_china_pmi",
        "macro_china_cpi",
        "macro_china_ppi",
        "macro_china_new_financial_credit",
        "macro_china_gdzctz",
    }:
        return _col_with_keywords(columns, ["月"])
    if spec.source == "macro_china_lpr":
        return _col_with_keywords(columns, ["TRADE_DATE"])
    if spec.source == "macro_china_reserve_requirement_ratio":
        return _col_with_keywords(columns, ["生效时间"])
    return _col_with_keywords(columns, ["日期"])


def build_series(spec: SeriesSpec) -> pd.Series:
    df = fetch_dataset(spec.source)
    columns = [str(c) for c in df.columns]
    date_col = resolve_date_col(spec, columns)
    value_col = _col_with_keywords(columns, spec.value_col.split("|"))

    out = df[[date_col, value_col]].copy()
    out.columns = ["Date", "Value"]
    out["Date"] = out["Date"].map(_to_timestamp)
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Date", "Value"]).sort_values("Date")
    out["AvailDate"] = _availability_date(out["Date"], spec.availability)
    out = out.groupby("AvailDate", as_index=True)["Value"].last().sort_index()
    out.name = f"{spec.group_prefix}_{spec.name}"
    return out


def specs() -> list[SeriesSpec]:
    return [
        SeriesSpec("pmi_mfg_level", "1", "macro_china_pmi", "制造业|指数", availability="pmi_month_end"),
        SeriesSpec("pmi_mfg_yoy", "1", "macro_china_pmi", "制造业|同比增长", availability="pmi_month_end"),
        SeriesSpec("pmi_non_mfg_level", "1", "macro_china_non_man_pmi", "今值", availability="pmi_month_end"),
        SeriesSpec("industrial_prod_yoy", "1", "macro_china_industrial_production_yoy", "今值", availability="next_month_15d"),
        SeriesSpec("fixed_asset_invest_cum", "1", "macro_china_gdzctz", "自年初累计", availability="next_month_15d"),
        SeriesSpec("exports_yoy", "1", "macro_china_exports_yoy", "今值", availability="next_month_12d"),
        SeriesSpec("cpi_yoy", "2", "macro_china_cpi", "全国|同比增长", availability="next_month_10d"),
        SeriesSpec("cpi_mom", "2", "macro_china_cpi", "全国|环比增长", availability="next_month_10d"),
        SeriesSpec("ppi_yoy", "2", "macro_china_ppi", "当月|同比增长", availability="next_month_10d"),
        SeriesSpec("m2_yoy", "3", "macro_china_m2_yearly", "今值", availability="next_month_12d"),
        SeriesSpec("new_credit_monthly", "3", "macro_china_new_financial_credit", "当月", availability="next_month_12d"),
        SeriesSpec("new_credit_yoy", "3", "macro_china_new_financial_credit", "当月|同比增长", availability="next_month_12d"),
        SeriesSpec("lpr_1y", "3", "macro_china_lpr", "LPR1Y", availability="same_day"),
        SeriesSpec("shibor_on", "4", "macro_china_shibor_all", "O/N|定价", availability="same_day"),
        SeriesSpec("shibor_1w", "4", "macro_china_shibor_all", "1W|定价", availability="same_day"),
        SeriesSpec("shibor_1m", "4", "macro_china_shibor_all", "1M|定价", availability="same_day"),
        SeriesSpec("shibor_3m", "4", "macro_china_shibor_all", "3M|定价", availability="same_day"),
        SeriesSpec("rrr_large_after", "4", "macro_china_reserve_requirement_ratio", "大型金融机构|调整后", availability="same_day"),
    ]


def fetch_csi_gov_bond_510y_active_close() -> pd.Series:
    """
    做多参考：中证 5–10 年期国债活跃券指数（CSI 931018）收盘。
    说明：该指数为「国债」口径（不含中债指数序列），样本为交易所国债活跃券，
    久期落在 5–10 年区间，用作中长端国债暴露的代理；与「恰好 7–10 年」字面区间
    可能略有差异，但较贴近实务中对「中长期国债」的刻画。
    """
    import akshare as ak

    end = pd.Timestamp.today().strftime("%Y%m%d")
    df = ak.stock_zh_index_hist_csindex(symbol="931018", start_date="20100101", end_date=end)
    df["Date"] = pd.to_datetime(df["日期"])
    s = df.set_index("Date")["收盘"].astype(float).sort_index()
    s.name = "5_long_gov_510y_csi931018_close"
    return s


def fetch_csi_short_term_paper_close() -> pd.Series:
    """
    做空参考：中证短融指数（H11014）收盘点位。
    说明：指数本身为「多头指数」；在多空框架中通常作为短端/信用短融的空头或对冲标的。
    """
    import akshare as ak

    end = pd.Timestamp.today().strftime("%Y%m%d")
    df = ak.stock_zh_index_hist_csindex(symbol="H11014", start_date="20100101", end_date=end)
    df["Date"] = pd.to_datetime(df["日期"])
    s = df.set_index("Date")["收盘"].astype(float).sort_index()
    s.name = "5_short_cp_csi_close"
    return s


def build_factor_panel(price_dates: pd.DatetimeIndex) -> pd.DataFrame:
    series_map: dict[str, pd.Series] = {}
    failures: list[str] = []
    for spec in specs():
        try:
            s = build_series(spec).reindex(price_dates).ffill()
            series_map[s.name] = s
            print(f"ok   {s.name:<28} rows={int(s.notna().sum())}")
        except Exception as e:
            failures.append(f"{spec.group_prefix}_{spec.name}: {e}")
            print(f"fail {spec.group_prefix}_{spec.name}: {e}")

    # --- 多空参考：中证国债活跃券（做多） vs 中证短融（做空/对冲参考），均为 CSI 口径、非中债 ---
    extra: list[tuple[str, str, object]] = [
        ("5_long_gov_510y_csi931018_close", "中证5-10年国债活跃券(931018)收盘", fetch_csi_gov_bond_510y_active_close),
        ("5_short_cp_csi_close", "中证短融(H11014)收盘", fetch_csi_short_term_paper_close),
    ]
    for col, label, fn in extra:
        try:
            s = fn().reindex(price_dates).ffill()
            series_map[col] = s
            print(f"ok   {col:<36} ({label}) rows={int(s.notna().sum())}")
        except Exception as e:
            failures.append(f"{col}: {e}")
            print(f"fail {col}: {e}")

    if not series_map:
        raise RuntimeError("No public macro series were fetched successfully")

    if failures:
        print("\nSome datasets failed and were skipped:")
        for msg in failures:
            print(" -", msg)

    factors = pd.DataFrame(series_map, index=price_dates)
    factors.index.name = "Date"
    factors = factors.sort_index().ffill(limit=90)

    long_s = factors.get("5_long_gov_510y_csi931018_close")
    short_s = factors.get("5_short_cp_csi_close")
    if long_s is not None and short_s is not None:
        rl = long_s.pct_change()
        rs = short_s.pct_change()
        # --- 1) 等额「多空」利差：+1 份国债指数收益 −1 份短融指数收益（日末再平衡、无费用），可视为简化对冲 ---
        ls_ret = (rl - rs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        factors["5_ls_gov510y_minus_shortcp_nav"] = (1.0 + ls_ret).cumprod()
        print("ok   5_ls_gov510y_minus_shortcp_nav           (1:1 dollar LS, daily rebalance, no cost)")

        # --- 2) 滚动 beta 对冲：用前一日估计的 beta 对冲短融，降低短融方向暴露（仍非保证金/滑点真实回测）---
        win = 60
        var_s = rs.rolling(win, min_periods=win // 2).var()
        cov_ls = rl.rolling(win, min_periods=win // 2).cov(rs)
        beta = (cov_ls / var_s.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        beta_lag = beta.shift(1)
        hedge_ret = (rl - beta_lag * rs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        factors["5_hedge_rollbeta60_ls_nav"] = (1.0 + hedge_ret).cumprod()
        print("ok   5_hedge_rollbeta60_ls_nav                (r_L - beta_{t-1}*r_S, rolling 60d beta)")

        # --- 3) 报告用策略净值：纯多 vs 多空（多空与 5_ls_* 等价，便于报告列名）---
        lv0 = float(long_s.dropna().iloc[0])
        factors["5_strat_longonly_nav"] = long_s / lv0
        factors["5_strat_longshort_nav"] = factors["5_ls_gov510y_minus_shortcp_nav"]
        print("ok   5_strat_longonly_nav                   (LongOnly: buy&hold 931018, norm=1)")
        print("ok   5_strat_longshort_nav                   (LongShort: 1:1 rL-rS, same as 5_ls_)")

        # --- 4) 切换策略（按信号在「国债」与「短融」之间二选一持有）---
        # 需求：
        # - 多信号：做多国债、平仓短融
        # - 空信号：做多短融、平仓国债
        # - 每天最多一次多/空信号（在日频上等价于：每个日期最多一次换仓）
        # - 每次换仓滑点：固定扣 0.01%
        #   注：这里用“收益扣减”口径，0.01% = 1bp = 0.0001
        switch_slip = 0.0001
        switch_sig = pd.Series(0.0, index=factors.index)
        # 使用 rL - rS 的符号作为“多/空”信号：>0 偏国债，<0 偏短融
        diff = (rl - rs).replace([np.inf, -np.inf], np.nan)
        switch_sig[diff > 0] = 1.0
        switch_sig[diff < 0] = -1.0
        # 若当天 diff==0 或不可得，则不发信号（保持空/前值由持仓逻辑决定）
        factors["5_sig_switch_gov_vs_cp"] = switch_sig

        # 持仓：多信号持有国债；空信号持有短融；0 信号则延续上一日持仓（避免无意义来回）
        # 用“持仓状态”统一驱动，避免 ffill 把两边都锁死的问题
        state = switch_sig.replace(0.0, np.nan).ffill().fillna(0.0)
        w_gov = (state == 1.0).astype(float)
        w_cp = (state == -1.0).astype(float)
        factors["5_pos_switch_gov_weight"] = w_gov
        factors["5_pos_switch_cp_weight"] = w_cp

        # 策略收益（次日执行）：用“已执行仓位”乘以当日收益
        w_gov_exe = w_gov.shift(1).fillna(0.0)
        w_cp_exe = w_cp.shift(1).fillna(0.0)
        strat_ret = w_gov_exe * rl.fillna(0.0) + w_cp_exe * rs.fillna(0.0)

        # 换仓成本：按“已执行仓位”的变化扣（与收益口径对齐），每个交易日最多一次
        traded = ((w_gov_exe.diff().abs() + w_cp_exe.diff().abs()) > 1e-12).astype(float).fillna(0.0)
        strat_ret = strat_ret - switch_slip * traded
        factors["5_strat_switch_gov_cp_nav"] = (1.0 + strat_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)).cumprod()
        print("ok   5_sig_switch_gov_vs_cp                 (+1 gov / -1 cp, at most once per day)")
        print("ok   5_strat_switch_gov_cp_nav              (switching, slippage=0.01% per trade-event)")

    return factors


def save_output(price: pd.DataFrame, factors: pd.DataFrame) -> None:
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", mode="w") as writer:
        price.to_excel(writer, sheet_name="Price", index=False)
        factors.reset_index().to_excel(writer, sheet_name="Factors", index=False)
    price.to_csv(OUT_PRICE_CSV, index=False, encoding="utf-8-sig")
    factors.reset_index().to_csv(OUT_FACTORS_CSV, index=False, encoding="utf-8-sig")


def main() -> None:
    price = attach_close_net_to_price(fetch_price())
    factors = build_factor_panel(pd.DatetimeIndex(price["Date"]))
    save_output(price, factors)
    print("saved", OUT_XLSX)
    print("saved", OUT_PRICE_CSV)
    print("saved", OUT_FACTORS_CSV)
    print("price rows:", len(price), "factor cols:", len(factors.columns))


if __name__ == "__main__":
    main()
