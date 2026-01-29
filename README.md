# causal_retail_analysis

A **simplified but econometrically rigorous** pipeline for evaluating localized retail interventions where **geography and spillovers matter**.

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

Right now, the pipeline assigns a **synthetic** treatment label (random \(50\%\)) so you can run end-to-end without intervention data; replace `treated` with your real assignment when available.

From the project root:

```bash
python -m causal_retail_analysis.analysis.pipeline
```

Optional parameters:

```bash
python -m causal_retail_analysis.analysis.pipeline --geo-eps-miles 37 --geo-min-samples 8 --treated-frac 0.5 --seed 42 --smd-threshold 0.2 --min-pairs 50
```

## Outputs to expect

- A recommended **spillover buffer** distance (miles) based on balance/feasibility
- A matched treated/control sample and **SMD balance diagnostics**
- An **ATE estimate** from IPW-weighted DiD with clustered SEs and a 95% CI
- Robustness summaries (sensitivity to weight trimming + placebo DiD)
