from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FAST_RESULTS = ROOT / "reproduce" / "public_fast_optimization_results.csv"
TARGET_RESULTS = ROOT / "reproduce" / "public_paper_target_optimization_results.csv"
OUT_RANK = ROOT / "reproduce" / "public_ranked_most_paper_like.csv"

TARGET_ANN = 0.037
TARGET_SHARPE = 1.27
TARGET_CALMAR = 1.12
TARGET_NONZERO = 0.55
TARGET_MDD = -0.033


def paper_like_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ann_gap_abs"] = (out["ann"] - TARGET_ANN).abs()
    out["sharpe_gap_abs"] = (out["sharpe"] - TARGET_SHARPE).abs()
    out["calmar_gap_abs"] = (out["calmar"] - TARGET_CALMAR).abs()
    out["nonzero_gap_abs"] = (out["nonzero_ratio"] - TARGET_NONZERO).abs()
    out["mdd_gap_abs"] = (out["mdd"] - TARGET_MDD).abs()
    out["paper_like_score"] = (
        -2.0 * out["ann_gap_abs"]
        -1.8 * out["sharpe_gap_abs"]
        -1.0 * out["calmar_gap_abs"]
        -0.6 * out["nonzero_gap_abs"]
        -1.2 * out["mdd_gap_abs"]
        + (out["ann"] > 0).astype(float) * 0.5
        + (out["sharpe"] > 0).astype(float) * 0.5
    )
    return out.sort_values(["paper_like_score", "ann_gap_abs", "sharpe_gap_abs"], ascending=[False, True, True])


def main() -> None:
    dfs = []
    if FAST_RESULTS.exists():
        fast = pd.read_csv(FAST_RESULTS)
        fast["source"] = "fast"
        dfs.append(fast)
    if TARGET_RESULTS.exists():
        target = pd.read_csv(TARGET_RESULTS)
        target["source"] = "paper_target"
        dfs.append(target)
    if not dfs:
        raise FileNotFoundError("Missing optimization result files. Run fast and paper-target optimization first.")

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    ranked = paper_like_score(merged)
    ranked.to_csv(OUT_RANK, index=False, encoding="utf-8-sig")
    print("saved", OUT_RANK)
    print(ranked.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
