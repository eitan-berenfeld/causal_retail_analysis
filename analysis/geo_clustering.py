import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import HDBSCAN
from matplotlib.patches import Circle
from causal_project import load_and_prepare_data

EARTH_RADIUS = 3959.0  # miles

# Haversine Distance calculation used for geo-spatial clustering
def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS * 2 * np.arcsin(np.sqrt(a))


# Run HDBSCAN
def run_hdbscan(df, min_cluster_size=8):
    coords = np.radians(df[['latitude', 'longitude']].values)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='haversine', cluster_selection_epsilon=0.05)
    labels = clusterer.fit_predict(coords)
    df['geo_cluster'] = labels
    return df, labels


# Evaluate clustering quality
def evaluate_clustering(df, labels):
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])

    print(f"Number of clusters (excluding noise): {n_clusters}")
    print("Cluster distribution:\n", pd.Series(labels).value_counts())

    # Silhouette score only on non-noise clusters
    valid = df[df['geo_cluster'] != -1]
    if len(valid.geo_cluster.unique()) > 1:
        coords = np.radians(valid[['latitude', 'longitude']].values)
        sil = silhouette_score(coords, valid['geo_cluster'], metric='haversine')
        print(f"Silhouette Score: {sil:.4f}")
    else:
        print("Silhouette score not computed (only one cluster).")


# Plot clusters with adaptive radius
def plot_clusters(df):
    def draw_cluster_circle(ax, cluster_id, percentile=90):
        subset = df[df['geo_cluster'] == cluster_id]
        if subset.empty:
            return

        center_lat = subset['latitude'].mean()
        center_lon = subset['longitude'].mean()

        dists = haversine_np(
            center_lat, center_lon,
            subset['latitude'].values,
            subset['longitude'].values
        )
        radius_miles = np.percentile(dists, percentile)
        radius_deg = radius_miles / 69

        circle = Circle(
            (center_lon, center_lat),
            radius=radius_deg,
            edgecolor='darkblue',
            facecolor='none',
            linestyle='--',
            linewidth=2
        )
        ax.add_patch(circle)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    unique_clusters = sorted([c for c in df['geo_cluster'].unique() if c != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for idx, cid in enumerate(unique_clusters):
        subset = df[df['geo_cluster'] == cid]
        ax.scatter(subset['longitude'], subset['latitude'],
                   s=30, color=colors[idx], alpha=0.7, label=f"Cluster {cid}")
        draw_cluster_circle(ax, cid)

    noise = df[df['geo_cluster'] == -1]
    if not noise.empty:
        ax.scatter(noise['longitude'], noise['latitude'], s=30, color='gray', alpha=0.4, label='Noise (-1)')

    plt.title("HDBSCAN Geo Clusters with Adaptive Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()


def geo_clustering_pipeline(merged_df, min_cluster_size=8):
    df = merged_df.dropna(subset=['latitude', 'longitude']).copy()

    # Run HDBSCAN (automatically determines clusters)
    print("\nRunning HDBSCAN...")
    df, labels = run_hdbscan(df, min_cluster_size=min_cluster_size)

    # Evaluate
    print("\nClustering Evaluation:")
    evaluate_clustering(df, labels)

    # Plot
    print("\nPlotting Clusters...")
    plot_clusters(df)

    # Return merged back to main df
    return merged_df.merge(df[['store_id', 'geo_cluster']], on='store_id', how='left')


if __name__ == "__main__":
    merged_df = load_and_prepare_data()
    result_df = geo_clustering_pipeline(merged_df, min_cluster_size=8)
    print("\nFinal result with geo clusters:\n", result_df.head(10))