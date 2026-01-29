from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DidIpwResult:
    """Container for IPW-weighted DiD regression results."""

    long_df: pd.DataFrame
    model: object
    ate: float
    ate_se: float
    ci95_low: float
    ci95_high: float


def _prep_covariates(df: pd.DataFrame, covariate_cols: Iterable[str]) -> pd.DataFrame:
    """Coerce covariates to numeric, mapping Yes/No to 1/0."""
    X = df[list(covariate_cols)].copy()
    for c in X.columns:
        if X[c].dtype == object:
            as_str = X[c].astype(str).str.strip().str.lower()
            mapped = as_str.map({"yes": 1, "no": 0})
            X[c] = mapped.where(mapped.notna(), X[c]).infer_objects(copy=False)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def fit_ipw_did(
    matched_df: pd.DataFrame,
    outcome_pre: str = "sales_pre",
    outcome_post: str = "sales_post",
    treated_col: str = "treated",
    unit_id_col: str = "store_id",
    covariate_cols: Iterable[str] = (
        "Discount_Rate",
        "Customer_Age",
        "Store_Size",
        "Inventory_Level",
        "Number_of_Employees",
        "Marketing_Spend",
        "Family",
        "Kids",
        "Weekend",
        "Holiday",
        "Foot_Traffic",
        "Average_Transaction_Value",
        "Online_Sales",
    ),
    weight_clip: tuple[float, float] = (1e-3, 1 - 1e-3),
) -> DidIpwResult:
    """Estimate an ATE using IPW-weighted two-period DiD with clustered standard errors."""
    req = [outcome_pre, outcome_post, treated_col, unit_id_col]
    missing = [c for c in req if c not in matched_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = _prep_covariates(matched_df, covariate_cols)
    y = matched_df[treated_col].astype(int)

    X_scaled = StandardScaler().fit_transform(X)
    ps_model = LogisticRegression(max_iter=2000)
    ps_model.fit(X_scaled, y)
    p = ps_model.predict_proba(X_scaled)[:, 1]
    p = np.clip(p, weight_clip[0], weight_clip[1])

    base = matched_df[[unit_id_col, treated_col, outcome_pre, outcome_post]].copy()
    base["propensity_score"] = p

    long_df = pd.concat(
        [
            pd.DataFrame(
                {
                    unit_id_col: base[unit_id_col],
                    "y": base[outcome_pre],
                    treated_col: base[treated_col],
                    "post": 0,
                    "propensity_score": base["propensity_score"],
                }
            ),
            pd.DataFrame(
                {
                    unit_id_col: base[unit_id_col],
                    "y": base[outcome_post],
                    treated_col: base[treated_col],
                    "post": 1,
                    "propensity_score": base["propensity_score"],
                }
            ),
        ],
        ignore_index=True,
    )

    def ipw(row) -> float:
        pp = float(row["propensity_score"])
        return 1.0 / pp if int(row[treated_col]) == 1 else 1.0 / (1.0 - pp)

    long_df["ipw"] = long_df.apply(ipw, axis=1)

    model = smf.wls(f"y ~ {treated_col} * post", data=long_df, weights=long_df["ipw"]).fit(
        cov_type="cluster",
        cov_kwds={"groups": long_df[unit_id_col]},
    )

    ate = float(model.params[f"{treated_col}:post"])
    se = float(model.bse[f"{treated_col}:post"])
    ci_low, ci_high = ate - 1.96 * se, ate + 1.96 * se

    return DidIpwResult(long_df=long_df, model=model, ate=ate, ate_se=se, ci95_low=float(ci_low), ci95_high=float(ci_high))

