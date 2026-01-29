from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .buffer_selection import choose_spatial_buffer
from .causal_project import load_and_prepare_data
from .covariate_clustering import add_covariate_clusters
from .did_ipw import fit_ipw_did
from .geo_clustering import geo_clustering_pipeline
from .matching import MatchingConfig, compute_smds, match_treated_controls
from .robustness import ipw_trim_sensitivity, placebo_outcome_did


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end simplified causal pipeline."""

    geo_eps_miles: float = 37.0
    geo_min_samples: int = 8
    cov_k_min: int = 2
    cov_k_max: int = 10
    treated_frac: float = 0.5
    random_state: int = 42
    smd_threshold: float = 0.2
    min_pairs: int = 50


DEFAULT_COVARIATES = [
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
]


def assign_treatment_demo(df: pd.DataFrame, treated_frac: float, random_state: int) -> pd.DataFrame:
    """Assign a synthetic treatment flag for demonstration when real treatment labels are unavailable."""
    if not (0.0 < treated_frac < 1.0):
        raise ValueError("treated_frac must be between 0 and 1")
    out = df.copy()
    rng = np.random.default_rng(random_state)
    treated_idx = rng.choice(out.index.to_numpy(), size=int(round(len(out) * treated_frac)), replace=False)
    out["treated"] = 0
    out.loc[treated_idx, "treated"] = 1
    return out


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Run the simplified geospatial causal inference pipeline and print key results."""
    merged_df = load_and_prepare_data()
    merged_df = assign_treatment_demo(merged_df, treated_frac=cfg.treated_frac, random_state=cfg.random_state)

    # 1) Geo clusters
    merged_df = geo_clustering_pipeline(
        merged_df,
        eps_miles=cfg.geo_eps_miles,
        min_samples=cfg.geo_min_samples,
        plot=False,
    )

    # 2) Covariate clusters
    cc = add_covariate_clusters(
        merged_df,
        covariate_cols=DEFAULT_COVARIATES,
        k_min=cfg.cov_k_min,
        k_max=cfg.cov_k_max,
        random_state=cfg.random_state,
        out_col="cov_cluster",
    )
    merged_df = cc.df
    print(f"\nCovariate clustering: selected k={cc.k} (silhouette={cc.silhouette:.3f})")

    # 3) Choose spatial buffer
    buf = choose_spatial_buffer(
        merged_df,
        treated_col="treated",
        geo_cluster_col="geo_cluster",
        cov_cluster_col="cov_cluster",
        covariate_cols_for_smd=("Store_Size", "Foot_Traffic", "Inventory_Level"),
        smd_threshold=cfg.smd_threshold,
        min_pairs=cfg.min_pairs,
        random_state=cfg.random_state,
    )
    print("\nBuffer sweep (head):")
    print(buf.results.head(10))
    print(f"\nRecommended buffer (miles): {buf.recommended_buffer_miles}")

    # 4) Match
    match_cfg = MatchingConfig(
        spatial_buffer_miles=buf.recommended_buffer_miles,
        allowed_cov_cluster_diff=0,
        no_reuse_controls=True,
        random_state=cfg.random_state,
    )
    mr = match_treated_controls(
        merged_df,
        treated_col="treated",
        geo_cluster_col="geo_cluster",
        cov_cluster_col="cov_cluster",
        config=match_cfg,
    )
    print(f"\nMatched pairs: {len(mr.pairs)}")

    smds = compute_smds(mr.matched_df, treated_col="treated", covariate_cols=DEFAULT_COVARIATES)
    print("\nTop SMDs (worst balance):")
    print(smds.head(10))

    # 5) IPW + DiD (2-period)
    did = fit_ipw_did(
        mr.matched_df,
        outcome_pre="sales_pre",
        outcome_post="sales_post",
        treated_col="treated",
        unit_id_col="store_id",
        covariate_cols=DEFAULT_COVARIATES,
    )
    print("\nIPW + DiD ATE (sales):")
    print(f"ATE={did.ate:.4f}  SE={did.ate_se:.4f}  95% CI=[{did.ci95_low:.4f}, {did.ci95_high:.4f}]")

    # 6) Minimal robustness
    sens = ipw_trim_sensitivity(did.long_df, treated_col="treated", unit_id_col="store_id", outcome_col="y")
    print("\nIPW trimming sensitivity:")
    print(sens.summary)

    try:
        placebo = placebo_outcome_did(mr.matched_df, placebo_pre="orders_pre", placebo_post="orders_post")
        print("\nPlacebo DiD (orders):")
        print(placebo.summary().tables[1])
    except ValueError as e:
        print(f"\nPlacebo DiD skipped: {e}")


def _parse_args() -> PipelineConfig:
    """Parse CLI args for running the pipeline as a script."""
    p = argparse.ArgumentParser(description="Run simplified geo-spatial causal retail pipeline.")
    p.add_argument("--geo-eps-miles", type=float, default=37.0)
    p.add_argument("--geo-min-samples", type=int, default=8)
    p.add_argument("--treated-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smd-threshold", type=float, default=0.2)
    p.add_argument("--min-pairs", type=int, default=50)
    args = p.parse_args()
    return PipelineConfig(
        geo_eps_miles=args.geo_eps_miles,
        geo_min_samples=args.geo_min_samples,
        treated_frac=args.treated_frac,
        random_state=args.seed,
        smd_threshold=args.smd_threshold,
        min_pairs=args.min_pairs,
    )


if __name__ == "__main__":
    run_pipeline(_parse_args())

