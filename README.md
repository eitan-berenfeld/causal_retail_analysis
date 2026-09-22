# Causal Retail Analysis 

A pipeline for evaluating localized retail interventions where geography and spillovers matter.

The problem this is built for: most naive before/after retail analyses ignore geography entirely, which breaks down the moment stores are close enough to affect each other. A promotion at one location can pull customers away from a "control" store two miles away, contaminating the comparison. The pipeline below treats that as a first-class problem instead of an edge case: it clusters stores geographically, picks a spillover buffer before matching anything, and only then estimates the effect.

## What this project does

Given store-level covariates, store locations (lat/lon), and outcomes measured **pre vs post** intervention, the pipeline:

- **Loads and merges data** into a single store-level table (`analysis/causal_project.py`)
- **Creates geo clusters** from latitude/longitude (DBSCAN with haversine distance) (`analysis/geo_clustering.py`)
- **Creates covariate clusters** to match like-with-like stores (PCA + KMeans) (`analysis/covariate_clustering.py`)
- **Chooses a spatial buffer** (in miles) to reduce spillovers while keeping a feasible sample (`analysis/buffer_selection.py`)
- **Matches treated stores to controls** with a geo-aware matching rule (`analysis/matching.py`)
- **Estimates the effect with IPW + DiD** using clustered standard errors (`analysis/did_ipw.py`)
- **Runs minimal robustness checks** (IPW trimming sensitivity + placebo outcome DiD) (`analysis/robustness.py`)

## How to run (current demo mode)

**There is no real intervention in this repo.** `assign_treatment_demo()` in `analysis/pipeline.py` assigns the `treated` flag at random to 50% of stores, so what runs end to end is the *machinery* — clustering, buffer selection, matching, IPW-weighted DiD — exercised against a known-null assignment. Any ATE it reports should be indistinguishable from zero; that is the point of running it this way, not a finding. Swap `assign_treatment_demo` for a real treated-store list and treatment date to get an estimate that means anything.

From the project root:

```bash
python -m analysis.pipeline
```

Optional parameters:

```bash
python -m analysis.pipeline --geo-eps-miles 37 --geo-min-samples 8 --treated-frac 0.5 --seed 42 --smd-threshold 0.2 --min-pairs 50
```

The sales outcome is pulled from the HuggingFace dataset `Dingdong-Inc/FreshRetailNet-50K` at runtime, so the first run needs network access.

## Outputs to expect

- A recommended **spillover buffer** distance (miles) based on balance/feasibility
- A matched treated/control sample and **SMD balance diagnostics**
- An **ATE estimate** from IPW-weighted DiD with clustered SEs and a 95% CI
- Robustness summaries (sensitivity to weight trimming + placebo DiD)


## Known defects

Stated plainly, because they bound what this repo currently demonstrates:

- **The three source datasets are joined by row position, not by a shared key.** Sales come from `Dingdong-Inc/FreshRetailNet-50K`, store covariates from `data/Store_Dataset.csv` (499 rows), and coordinates from `data/store_location_dataset.csv` (4,654 rows). The merge is `pd.merge(agg_metrics, store_df, left_on='store_id', right_index=True)`, so a store's covariates are whatever row happened to sit at that index in an unrelated file, and any `store_id` at or above 499 is silently dropped. The matching and propensity steps are therefore running on covariates that do not belong to the outcome units. This is the first thing to fix.
- **The pre/post split is arbitrary.** `treatment_date = 2024-05-10` is hardcoded with no event behind it.
- **No tests.** CI runs pylint only.
- **Requirements are unpinned**, so a clean install is not reproducible across time.

## Where this goes next

Replace the demo assignment with a real treatment definition (a treated-store list plus a treatment date), join the sources on a genuine key, and extend the DiD panel to multiple pre/post periods — which is what unlocks the robustness checks that matter here, parallel-trends and an event study.