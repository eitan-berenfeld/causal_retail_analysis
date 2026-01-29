from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class WeightTrimSensitivityResult:
    """Compare ATE estimates under different IPW trimming rules."""

    summary: pd.DataFrame


def ipw_trim_sensitivity(
    long_df: pd.DataFrame,
    treated_col: str = "treated",
    unit_id_col: str = "store_id",
    outcome_col: str = "y",
    trims: Iterable[float] = (0.0, 0.01, 0.02, 0.05),
) -> WeightTrimSensitivityResult:
    """Re-fit weighted DiD after trimming extreme IPW weights at different quantiles."""
    required = [treated_col, unit_id_col, outcome_col, "post", "ipw"]
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[dict[str, float]] = []
    for q in trims:
        q = float(q)
        if not (0.0 <= q < 0.5):
            raise ValueError("Each trim must satisfy 0 <= trim < 0.5")

        if q == 0.0:
            df = long_df.copy()
        else:
            lo = long_df["ipw"].quantile(q)
            hi = long_df["ipw"].quantile(1 - q)
            df = long_df[(long_df["ipw"] >= lo) & (long_df["ipw"] <= hi)].copy()

        model = smf.wls(f"{outcome_col} ~ {treated_col} * post", data=df, weights=df["ipw"]).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[unit_id_col]},
        )
        ate = float(model.params[f"{treated_col}:post"])
        se = float(model.bse[f"{treated_col}:post"])
        rows.append(
            {
                "trim_quantile": q,
                "n_obs": int(len(df)),
                "ate": ate,
                "ate_se": se,
                "ci95_low": ate - 1.96 * se,
                "ci95_high": ate + 1.96 * se,
            }
        )

    return WeightTrimSensitivityResult(summary=pd.DataFrame(rows).sort_values("trim_quantile"))


def placebo_outcome_did(
    matched_df: pd.DataFrame,
    treated_col: str = "treated",
    unit_id_col: str = "store_id",
    placebo_pre: str = "orders_pre",
    placebo_post: str = "orders_post",
) -> object:
    """Run an unweighted DiD on a placebo outcome as a basic falsification check."""
    req = [treated_col, unit_id_col, placebo_pre, placebo_post]
    missing = [c for c in req if c not in matched_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    long_df = pd.concat(
        [
            pd.DataFrame(
                {
                    unit_id_col: matched_df[unit_id_col],
                    "y": matched_df[placebo_pre],
                    treated_col: matched_df[treated_col],
                    "post": 0,
                }
            ),
            pd.DataFrame(
                {
                    unit_id_col: matched_df[unit_id_col],
                    "y": matched_df[placebo_post],
                    treated_col: matched_df[treated_col],
                    "post": 1,
                }
            ),
        ],
        ignore_index=True,
    )
    # Make formula evaluation robust to odd dtypes (e.g., object arrays).
    long_df["y"] = pd.to_numeric(long_df["y"], errors="coerce").astype(float)
    long_df[treated_col] = pd.to_numeric(long_df[treated_col], errors="coerce").astype(int)
    long_df["post"] = pd.to_numeric(long_df["post"], errors="coerce").astype(int)
    long_df = long_df.dropna(subset=["y"])
    if long_df.empty:
        raise ValueError("Placebo outcome has no non-missing observations after coercion.")
    return smf.ols(f"y ~ {treated_col} * post", data=long_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": long_df[unit_id_col]},
    )

