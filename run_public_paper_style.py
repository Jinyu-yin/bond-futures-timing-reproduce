from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from paper_style_bond_futures_timing import Config, run


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INPUT = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_CSV = ROOT / "reproduce" / "paper_style_nav_public.csv"


def main() -> None:
    if not PUBLIC_INPUT.exists():
        raise FileNotFoundError(f"Missing {PUBLIC_INPUT}. Run build_public_macro_dataset.py first.")

    cfg = Config(input_xlsx=PUBLIC_INPUT, out_csv=OUT_CSV)
    df = run(cfg)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("saved", OUT_CSV, "rows", len(df))
    print(df[["Date", "StrategyNav", "BenchNav", "FinalPos"]].tail().to_string(index=False))


if __name__ == "__main__":
    main()
