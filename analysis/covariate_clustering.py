from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CovariateClusteringResult:
    """Container for covariate clustering outputs."""

    df: pd.DataFrame
    covariate_cols: list[str]
    k: int
    silhouette: float


def _yes_no_to_int(s: pd.Series) -> pd.Series:
    """Map common Yes/No strings to 1/0."""
    if s.dtype != object:
        return s

    out = s.copy()
    # Handle booleans explicitly.
    out = out.map({True: 1, False: 0}).fillna(out)

    # Handle common Yes/No strings without pandas `replace` downcasting.
    as_str = out.astype(str).str.strip().str.lower()
    mapped = as_str.map({"yes": 1, "no": 0})
    out = mapped.where(mapped.notna(), out)
    return out.infer_objects(copy=False)


def add_covariate_clusters(
    df: pd.DataFrame,
    covariate_cols: Iterable[str],
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    out_col: str = "cov_cluster",
) -> CovariateClusteringResult:
    """Cluster stores by covariates using standardized PCA(2) + KMeans with silhouette-based k selection."""
    covariate_cols = list(covariate_cols)
    if not covariate_cols:
        raise ValueError("covariate_cols must be non-empty")
    if k_min < 2 or k_max < k_min:
        raise ValueError("Require 2 <= k_min <= k_max")

    X = df[covariate_cols].copy()
    for c in covariate_cols:
        if X[c].dtype == object:
            X[c] = _yes_no_to_int(X[c])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

    best_k = None
    best_sil = -np.inf
    best_labels = None

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = model.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    out = df.copy()
    out[out_col] = best_labels
    return CovariateClusteringResult(df=out, covariate_cols=covariate_cols, k=int(best_k), silhouette=float(best_sil))

