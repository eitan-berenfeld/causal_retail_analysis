from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from .causal_project import load_and_prepare_data

EARTH_RADIUS_MILES = 3959.0


def haversine_np(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Compute great-circle distance in miles (vectorized)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(a))


def run_dbscan_haversine(
    df: pd.DataFrame,
    eps_miles: float,
    min_samples: int = 8,
    out_col: str = "geo_cluster",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Cluster points with DBSCAN using haversine distance on lat/lon."""
    if eps_miles <= 0:
        raise ValueError("eps_miles must be > 0")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")

    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    eps_radians = eps_miles / EARTH_RADIUS_MILES
    model = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(coords)
    out = df.copy()
    out[out_col] = labels
    return out, labels


# Evaluate clustering quality
def evaluate_clustering(df: pd.DataFrame, labels: np.ndarray, cluster_col: str = "geo_cluster") -> None:
    """Print basic cluster counts and (optional) haversine silhouette score."""
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])

    print(f"Number of clusters (excluding noise): {n_clusters}")
    print("Cluster distribution:\n", pd.Series(labels).value_counts())

    valid = df[df[cluster_col] != -1]
    if len(valid[cluster_col].unique()) > 1:
        coords = np.radians(valid[["latitude", "longitude"]].to_numpy())
        sil = silhouette_score(coords, valid[cluster_col], metric="haversine")
        print(f"Silhouette Score: {sil:.4f}")
    else:
        print("Silhouette score not computed (needs >= 2 clusters).")


# Plot clusters with adaptive radius
def plot_clusters(df: pd.DataFrame, cluster_col: str = "geo_cluster") -> None:
    """Plot geo clusters and a simple adaptive-radius boundary per cluster."""

    def draw_cluster_circle(ax, cluster_id: int, percentile: int = 90) -> None:
        subset = df[df[cluster_col] == cluster_id]
        if subset.empty:
            return

        center_lat = subset["latitude"].mean()
        center_lon = subset["longitude"].mean()

        dists = haversine_np(
            center_lat,
            center_lon,
            subset["latitude"].to_numpy(),
            subset["longitude"].to_numpy(),
        )
        radius_miles = np.percentile(dists, percentile)
        radius_deg = radius_miles / 69  # rough conversion (miles -> degrees lat)

        circle = Circle(
            (center_lon, center_lat),
            radius=radius_deg,
            edgecolor="darkblue",
            facecolor="none",
            linestyle="--",
            linewidth=2,
        )
        ax.add_patch(circle)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    unique_clusters = sorted([c for c in df[cluster_col].unique() if c != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_clusters))))

    for idx, cid in enumerate(unique_clusters):
        subset = df[df[cluster_col] == cid]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            s=30,
            color=colors[idx],
            alpha=0.7,
            label=f"Cluster {cid}",
        )
        draw_cluster_circle(ax, cid)

    noise = df[df[cluster_col] == -1]
    if not noise.empty:
        ax.scatter(
            noise["longitude"],
            noise["latitude"],
            s=30,
            color="gray",
            alpha=0.4,
            label="Noise (-1)",
        )

    plt.title("DBSCAN Geo Clusters (Haversine)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def geo_clustering_pipeline(
    merged_df: pd.DataFrame,
    eps_miles: float = 37.0,
    min_samples: int = 8,
    plot: bool = True,
) -> pd.DataFrame:
    """Add `geo_cluster` labels to `merged_df` using DBSCAN-haversine."""
    df = merged_df.dropna(subset=["latitude", "longitude"]).copy()

    print("\nRunning DBSCAN-haversine with parameters:")
    print(f"  eps_miles={eps_miles}")
    print(f"  min_samples={min_samples}")

    df, labels = run_dbscan_haversine(df, eps_miles=eps_miles, min_samples=min_samples)

    print("\nClustering Evaluation:")
    evaluate_clustering(df, labels, cluster_col="geo_cluster")

    if plot:
        print("\nPlotting Clusters...")
        plot_clusters(df, cluster_col="geo_cluster")

    return merged_df.merge(df[["store_id", "geo_cluster"]], on="store_id", how="left")


if __name__ == "__main__":
    merged_df = load_and_prepare_data()
    result_df = geo_clustering_pipeline(merged_df, eps_miles=37.0, min_samples=8, plot=True)
    print("\nFinal result with geo clusters:\n", result_df.head(10))