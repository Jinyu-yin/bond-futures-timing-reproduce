from __future__ import annotations

from pathlib import Path

import pandas as pd

from paper_style_bond_futures_timing import Config, run


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INPUT = ROOT / "reproduce" / "public_input_data.xlsx"
OUT_CSV_0 = ROOT / "reproduce" / "paper_style_nav_public.csv"
OUT_CSV_1 = ROOT / "reproduce" / "paper_style_nav_public_th1.csv"


def run_one(open_threshold: int, out_csv: Path) -> pd.DataFrame:
    cfg = Config(input_xlsx=PUBLIC_INPUT, out_csv=out_csv, open_threshold=open_threshold)
    df = run(cfg)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    if not PUBLIC_INPUT.exists():
        raise FileNotFoundError(f"Missing {PUBLIC_INPUT}. Run build_public_macro_dataset.py first.")

    df0 = run_one(open_threshold=0, out_csv=OUT_CSV_0)
    df1 = run_one(open_threshold=1, out_csv=OUT_CSV_1)

    print("saved", OUT_CSV_0, "rows", len(df0))
    print("saved", OUT_CSV_1, "rows", len(df1))
    print("threshold=0 tail")
    print(df0[["Date", "StrategyNav", "BenchNav", "FinalPos"]].tail().to_string(index=False))
    print("threshold=1 tail")
    print(df1[["Date", "StrategyNav", "BenchNav", "FinalPos"]].tail().to_string(index=False))


if __name__ == "__main__":
    main()
