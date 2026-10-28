"""
对比两类诉求：
  1) 放宽出手：open_threshold=0（有方向就开仓） vs 1（需 |GroupSum|>1）
  2) 净价 + 不对称杠杆：price_close_col=CloseNet，long_weight=1.2，short_weight=0.8，可保留 threshold=1

依赖：先 py reproduce/build_public_macro_dataset.py（写入 CloseNet）。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from paper_style_bond_futures_timing import Config, directional_win_rates, run


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INPUT = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_DIR = ROOT / "reproduce"


def _fmt_wr(wr: dict[str, float]) -> str:
    def q(x: float) -> str:
        return f"{x:.4f}" if pd.notna(x) and abs(x) < 1e100 else "nan"

    return (
        f"L={q(wr['long_win_rate'])} S={q(wr['short_win_rate'])} "
        f"allInv={q(wr['invested_win_rate'])} "
        f"(nL={int(wr['n_long_days'])}, nS={int(wr['n_short_days'])})"
    )


def main() -> None:
    if not PUBLIC_INPUT.exists():
        raise FileNotFoundError(f"Missing {PUBLIC_INPUT}. Run build_public_macro_dataset.py first.")

    price = pd.read_excel(PUBLIC_INPUT, sheet_name="Price")
    has_net = "CloseNet" in price.columns and pd.to_numeric(price["CloseNet"], errors="coerce").notna().any()

    scenarios: list[tuple[str, Config]] = []

    def add(name: str, **kw: object) -> None:
        out_csv = OUT_DIR / f"paper_style_nav_public_exp_{name}.csv"
        base = Config(input_xlsx=PUBLIC_INPUT, out_csv=out_csv)
        scenarios.append((name, replace(base, **kw)))

    add("relax_th0", open_threshold=0, price_close_col="Close")
    add("strict_th1", open_threshold=1, price_close_col="Close")
    if has_net:
        add("net_strict_th1_lev12_08", open_threshold=1, price_close_col="CloseNet", long_weight=1.2, short_weight=0.8)
        add("net_relax_th0_lev12_08", open_threshold=0, price_close_col="CloseNet", long_weight=1.2, short_weight=0.8)
    else:
        print("skip CloseNet scenarios (no CloseNet column); rebuild public_input_data.xlsx")

    rows: list[dict[str, object]] = []
    for name, cfg in scenarios:
        df = run(cfg)
        cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.out_csv, index=False, encoding="utf-8-sig")
        wr = directional_win_rates(df["Ret"], df["FinalPos"])
        last_nav = float(df["StrategyNav"].iloc[-1])
        row = {"scenario": name, "last_StrategyNav": last_nav, **wr}
        rows.append(row)
        print(f"{name:30} nav_end={last_nav:.6f}  {_fmt_wr(wr)}")

    summary = pd.DataFrame(rows)
    sum_path = OUT_DIR / "public_experiments_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8-sig")
    print("saved", sum_path)


if __name__ == "__main__":
    main()
