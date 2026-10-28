from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover
    GaussianHMM = None


@dataclass(frozen=True)
class HMMOverlayConfig:
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


def _safe_zscore(s: pd.Series) -> pd.Series:
    mu = float(s.mean()) if len(s) else 0.0
    sd = float(s.std()) if len(s) else 0.0
    if not np.isfinite(sd) or sd < 1e-8:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def build_hmm_features(
    signal_z: pd.Series,
    r_gov: pd.Series,
    r_cp: pd.Series,
    cfg: HMMOverlayConfig,
) -> pd.DataFrame:
    signal_z = pd.to_numeric(signal_z, errors="coerce").fillna(0.0)
    spread = pd.to_numeric(r_gov, errors="coerce").fillna(0.0) - pd.to_numeric(r_cp, errors="coerce").fillna(0.0)
    spread_vol = spread.rolling(cfg.vol_window, min_periods=max(5, cfg.vol_window // 2)).std().fillna(0.0)
    crowd = signal_z.abs().rolling(cfg.crowd_window, min_periods=max(5, cfg.crowd_window // 2)).mean().fillna(0.0)
    shock = signal_z.diff().abs().rolling(cfg.shock_window, min_periods=max(3, cfg.shock_window // 2)).mean().fillna(0.0)
    return pd.DataFrame(
        {
            "spread_ret": spread,
            "spread_vol": spread_vol,
            "signal_abs": signal_z.abs(),
            "signal_crowd": crowd,
            "signal_shock": shock,
        },
        index=signal_z.index,
    ).replace([np.inf, -np.inf], np.nan)


def _resolve_refit_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    scheduled = pd.date_range(start=index.min(), end=index.max(), freq=freq)
    if len(scheduled) == 0 or index.min() < scheduled[0]:
        scheduled = pd.DatetimeIndex([index.min()]).append(scheduled)
    actual: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for dt in scheduled:
        pos = index.searchsorted(dt, side="right") - 1
        if pos < 0:
            continue
        curr = pd.Timestamp(index[pos])
        if curr not in seen:
            actual.append(curr)
            seen.add(curr)
    if index.max() not in seen:
        actual.append(pd.Timestamp(index.max()))
    return pd.DatetimeIndex(actual).sort_values()


def compute_hmm_overlay(
    signal_z: pd.Series,
    r_gov: pd.Series,
    r_cp: pd.Series,
    cfg: HMMOverlayConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or HMMOverlayConfig()
    if GaussianHMM is None:
        raise ImportError("hmmlearn is required for HMM overlay")

    idx = pd.DatetimeIndex(signal_z.index).intersection(pd.DatetimeIndex(r_gov.index)).intersection(pd.DatetimeIndex(r_cp.index)).sort_values()
    signal_z = pd.to_numeric(signal_z.reindex(idx), errors="coerce").fillna(0.0)
    r_gov = pd.to_numeric(r_gov.reindex(idx), errors="coerce").fillna(0.0)
    r_cp = pd.to_numeric(r_cp.reindex(idx), errors="coerce").fillna(0.0)
    spread = r_gov - r_cp
    features = build_hmm_features(signal_z, r_gov, r_cp, cfg).reindex(idx)

    decision_rows: list[dict[str, float | pd.Timestamp]] = []
    for curr in _resolve_refit_dates(idx, cfg.refit_freq):
        hist = features.loc[:curr].dropna().tail(cfg.fit_window)
        scale = 1.0
        bias = 0.0
        score = 0.0
        state = np.nan

        if len(hist) >= cfg.min_history and hist.shape[1] >= 2:
            mu = hist.mean()
            sd = hist.std().replace(0.0, np.nan).fillna(1.0)
            x_hist = ((hist - mu) / sd).clip(-6.0, 6.0).to_numpy()
            try:
                model = GaussianHMM(
                    n_components=cfg.n_states,
                    covariance_type=cfg.covariance_type,
                    n_iter=cfg.n_iter,
                    random_state=cfg.random_state,
                )
                model.fit(x_hist)
                hidden = pd.Series(model.predict(x_hist), index=hist.index, dtype=float)
                stats = []
                spread_hist = spread.reindex(hist.index).fillna(0.0)
                for state_id in range(cfg.n_states):
                    mask = hidden == state_id
                    if not bool(mask.any()):
                        continue
                    stats.append(
                        {
                            "state": state_id,
                            "edge": float(spread_hist[mask].mean()),
                            "vol": float(hist.loc[mask, "spread_vol"].mean()),
                            "crowd": float(hist.loc[mask, "signal_crowd"].mean()),
                            "shock": float(hist.loc[mask, "signal_shock"].mean()),
                        }
                    )
                if stats:
                    stats_df = pd.DataFrame(stats).set_index("state")
                    stats_df["edge_z"] = _safe_zscore(stats_df["edge"])
                    stats_df["vol_z"] = _safe_zscore(stats_df["vol"])
                    stats_df["crowd_z"] = _safe_zscore(stats_df["crowd"])
                    stats_df["shock_z"] = _safe_zscore(stats_df["shock"])
                    stats_df["score"] = (
                        stats_df["edge_z"]
                        - 0.75 * stats_df["vol_z"]
                        - 0.50 * stats_df["crowd_z"]
                        - 0.25 * stats_df["shock_z"]
                    )
                    smin = float(stats_df["score"].min())
                    smax = float(stats_df["score"].max())
                    if smax - smin < 1e-8:
                        stats_df["score_norm"] = 0.0
                    else:
                        stats_df["score_norm"] = 2.0 * (stats_df["score"] - smin) / (smax - smin) - 1.0
                    state = float(hidden.iloc[-1])
                    score = float(stats_df.loc[int(state), "score_norm"])
                    scale = float(np.clip(1.0 + cfg.scale_strength * score, cfg.min_scale, cfg.max_scale))
                    bias = float(np.clip(cfg.bias_strength * score, -cfg.bias_strength, cfg.bias_strength))
            except Exception:
                scale = 1.0
                bias = 0.0
                score = 0.0
                state = np.nan

        decision_rows.append(
            {
                "Date": curr,
                "HMM_State": state,
                "HMM_Score": score,
                "HMM_Scale": scale,
                "HMM_Bias": bias,
            }
        )

    overlay = pd.DataFrame(decision_rows).set_index("Date").sort_index()
    overlay = overlay.reindex(idx).ffill()
    overlay["HMM_State"] = overlay["HMM_State"].astype(float)
    overlay["HMM_Score"] = overlay["HMM_Score"].fillna(0.0)
    overlay["HMM_Scale"] = overlay["HMM_Scale"].fillna(1.0)
    overlay["HMM_Bias"] = overlay["HMM_Bias"].fillna(0.0)
    overlay["Signal_Z_HMM"] = signal_z * overlay["HMM_Scale"] + overlay["HMM_Bias"]
    return overlay
