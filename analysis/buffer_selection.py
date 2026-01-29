from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .matching import MatchingConfig, compute_smds, match_treated_controls


@dataclass(frozen=True)
class BufferSelectionResult:
    """Summary of buffer sweep results and a recommended buffer."""

    results: pd.DataFrame
    recommended_buffer_miles: float


def choose_spatial_buffer(
    df: pd.DataFrame,
    buffers_miles: Iterable[float] = tuple(range(0, 51, 5)),
    treated_col: str = "treated",
    geo_cluster_col: str = "geo_cluster",
    cov_cluster_col: str = "cov_cluster",
    covariate_cols_for_smd: Iterable[str] = ("Store_Size", "Foot_Traffic", "Inventory_Level"),
    smd_threshold: float = 0.2,
    min_pairs: int = 25,
    random_state: int = 42,
) -> BufferSelectionResult:
    """Sweep buffer distances and pick the smallest buffer achieving balance and feasible sample size."""
    buffers = [float(b) for b in buffers_miles]
    covariate_cols_for_smd = list(covariate_cols_for_smd)

    rows: list[dict[str, float]] = []
    for b in buffers:
        cfg = MatchingConfig(
            spatial_buffer_miles=b,
            allowed_cov_cluster_diff=0,
            no_reuse_controls=True,
            random_state=random_state,
        )
        mr = match_treated_controls(
            df,
            treated_col=treated_col,
            geo_cluster_col=geo_cluster_col,
            cov_cluster_col=cov_cluster_col,
            config=cfg,
        )
        n_pairs = int(len(mr.pairs))
        if n_pairs == 0:
            rows.append(
                {
                    "buffer_miles": b,
                    "n_pairs": 0,
                    "max_smd": np.nan,
                }
            )
            continue

        smds = compute_smds(mr.matched_df, treated_col=treated_col, covariate_cols=covariate_cols_for_smd)
        rows.append(
            {
                "buffer_miles": b,
                "n_pairs": n_pairs,
                "max_smd": float(smds.max()),
            }
        )

    results = pd.DataFrame(rows).sort_values("buffer_miles").reset_index(drop=True)

    feasible = results[(results["n_pairs"] >= min_pairs) & (results["max_smd"] <= smd_threshold)]
    if not feasible.empty:
        recommended = float(feasible.sort_values(["buffer_miles", "max_smd"]).iloc[0]["buffer_miles"])
    else:
        # Fallback: prefer sample size, then balance.
        recommended = float(results.sort_values(["n_pairs", "max_smd"], ascending=[False, True]).iloc[0]["buffer_miles"])

    return BufferSelectionResult(results=results, recommended_buffer_miles=recommended)

