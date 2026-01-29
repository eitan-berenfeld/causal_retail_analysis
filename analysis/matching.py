from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .geo_clustering import haversine_np


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for geo-aware covariate-aware 1:1 matching."""

    spatial_buffer_miles: float = 25.0
    allowed_cov_cluster_diff: int = 0
    no_reuse_controls: bool = True
    random_state: int = 42


@dataclass(frozen=True)
class MatchResult:
    """Container for matched sample outputs."""

    matched_df: pd.DataFrame
    pairs: pd.DataFrame


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def match_treated_controls(
    df: pd.DataFrame,
    treated_col: str = "treated",
    geo_cluster_col: str = "geo_cluster",
    cov_cluster_col: str = "cov_cluster",
    config: MatchingConfig = MatchingConfig(),
) -> MatchResult:
    """Match treated stores to controls using covariate-cluster similarity plus a spatial spillover buffer."""
    _require_cols(df, [treated_col, "latitude", "longitude", geo_cluster_col, cov_cluster_col, "store_id"])
    if config.spatial_buffer_miles < 0:
        raise ValueError("spatial_buffer_miles must be >= 0")
    if config.allowed_cov_cluster_diff < 0:
        raise ValueError("allowed_cov_cluster_diff must be >= 0")

    rng = np.random.default_rng(config.random_state)

    treated_df = df[df[treated_col] == 1].copy()
    control_df = df[df[treated_col] == 0].copy()
    used_control_ids: set[int] = set()

    # Randomize treated order to reduce tie/path dependence.
    treated_order = treated_df.index.to_numpy()
    rng.shuffle(treated_order)

    pairs: list[dict[str, int | float]] = []

    for t_idx in treated_order:
        t = df.loc[t_idx]
        t_lat, t_lon = float(t["latitude"]), float(t["longitude"])
        t_geo = int(t[geo_cluster_col]) if pd.notna(t[geo_cluster_col]) else -1
        t_cov = int(t[cov_cluster_col])

        candidates = control_df.copy()
        if config.no_reuse_controls and used_control_ids:
            candidates = candidates[~candidates.index.isin(used_control_ids)]

        # Covariate-cluster similarity filter.
        candidates = candidates[(candidates[cov_cluster_col] - t_cov).abs() <= config.allowed_cov_cluster_diff]
        if candidates.empty:
            continue

        distances = haversine_np(
            t_lat,
            t_lon,
            candidates["latitude"].to_numpy(dtype=float),
            candidates["longitude"].to_numpy(dtype=float),
        )
        candidates = candidates.assign(distance_miles=distances)

        # Spillover rule: if same geo cluster, require distance >= buffer.
        if config.spatial_buffer_miles > 0:
            same_geo = candidates[geo_cluster_col] == t_geo
            ok = (~same_geo) | (candidates["distance_miles"] >= config.spatial_buffer_miles)
            candidates = candidates[ok]

        if candidates.empty:
            continue

        # Choose nearest remaining candidate.
        best = candidates.sort_values("distance_miles").iloc[0]
        c_idx = int(best.name)
        pairs.append(
            {
                "treated_index": int(t_idx),
                "control_index": c_idx,
                "distance_miles": float(best["distance_miles"]),
            }
        )
        used_control_ids.add(c_idx)

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return MatchResult(matched_df=df.iloc[0:0].copy(), pairs=pairs_df)

    treated_final = df.loc[pairs_df["treated_index"]].copy()
    control_final = df.loc[pairs_df["control_index"]].copy()
    treated_final = treated_final.assign(match_group="treated")
    control_final = control_final.assign(match_group="control")

    matched_df = pd.concat([treated_final, control_final], ignore_index=True)
    return MatchResult(matched_df=matched_df, pairs=pairs_df)


def compute_smds(
    matched_df: pd.DataFrame,
    treated_col: str,
    covariate_cols: Iterable[str],
) -> pd.Series:
    """Compute standardized mean differences (SMD) for treated vs control covariates."""
    covariate_cols = list(covariate_cols)
    _require_cols(matched_df, [treated_col, *covariate_cols])

    treated = matched_df[matched_df[treated_col] == 1]
    control = matched_df[matched_df[treated_col] == 0]

    smd: dict[str, float] = {}
    for cov in covariate_cols:
        t_raw = treated[cov]
        c_raw = control[cov]
        if t_raw.dtype == object:
            t_map = t_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
            t_raw = t_map.where(t_map.notna(), t_raw)
        if c_raw.dtype == object:
            c_map = c_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
            c_raw = c_map.where(c_map.notna(), c_raw)
        t = pd.to_numeric(t_raw, errors="coerce").astype(float)
        c = pd.to_numeric(c_raw, errors="coerce").astype(float)
        mt, mc = t.mean(), c.mean()
        pooled = np.sqrt((t.std(ddof=1) ** 2 + c.std(ddof=1) ** 2) / 2)
        smd[cov] = float(abs(mt - mc) / pooled) if pooled and not np.isnan(pooled) else 0.0

    return pd.Series(smd).sort_values(ascending=False)

