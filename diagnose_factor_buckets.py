from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_NAV = ROOT / "reproduce" / "paper_style_nav_public.csv"
FAST_NAV = ROOT / "reproduce" / "paper_style_nav_public_fast_best.csv"
TARGET_NAV = ROOT / "reproduce" / "paper_style_nav_public_paper_target_best.csv"
REFINE_NAV = ROOT / "reproduce" / "paper_style_nav_public_second_stage_best.csv"
OUT_CSV = ROOT / "reproduce" / "factor_bucket_diagnosis.csv"


def bucket_metrics(df: pd.DataFrame, label: str) -> list[dict[str, object]]:
    rows = []
    nav_cols = ["1_growth_nav", "2_inflation_nav", "3_credit_nav", "4_omo_nav"]
    sig_cols = ["1_growth_sig", "2_inflation_sig", "3_credit_sig", "4_omo_sig"]

    for nav_col, sig_col in zip(nav_cols, sig_cols):
        nav = df[nav_col]
        ret = nav.pct_change().fillna(0.0)
        ann = (1 + ret.mean()) ** 252 - 1
        vol = ret.std() * (252 ** 0.5)
        sharpe = ann / vol if vol > 1e-12 else None
        hit = float((ret > 0).mean())
        active = float((df[sig_col] != 0).mean())
        terminal = float(nav.iloc[-1])
        rows.append(
            {
                "strategy": label,
                "bucket": nav_col.replace("_nav", ""),
                "terminal_nav": terminal,
                "ann_proxy": float(ann),
                "vol_proxy": float(vol),
                "sharpe_proxy": float(sharpe) if sharpe is not None else None,
                "hit_ratio": hit,
                "active_ratio": active,
            }
        )
    return rows


def load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def main() -> None:
    datasets = {
        "base": load(BASE_NAV),
        "fast_best": load(FAST_NAV),
        "paper_target_best": load(TARGET_NAV),
        "second_stage_best": load(REFINE_NAV),
    }

    rows = []
    for label, df in datasets.items():
        if df is None:
            continue
        rows.extend(bucket_metrics(df, label))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("saved", OUT_CSV)
    print(out.sort_values(["strategy", "sharpe_proxy"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
