from __future__ import annotations
import os
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import HDBSCAN
from umap import UMAP
import json


def _iter_batches_permod(X: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield modality-specific batches: returns x0,x1,x2 each shaped (B,2000)."""
    N = X.shape[0]
    for start in range(0, N, batch_size):
        stop = min(N, start + batch_size)
        xb = X[start:stop].astype(np.float32)  # (B,3,5,20,20)
        
        # Split modalities and flatten last 3 dims
        x0 = xb[:, 0].reshape(xb.shape[0], -1)  # (B,2000)
        x1 = xb[:, 1].reshape(xb.shape[0], -1)
        x2 = xb[:, 2].reshape(xb.shape[0], -1)
        
        # Clean non-finite values
        for arr in (x0, x1, x2):
            arr[~np.isfinite(arr)] = 0.0
            
        yield x0, x1, x2


def run_permod_pca_umap(
    X: np.ndarray,
    ncomp_per_mod: int = 32,
    batch_size: int = 4096,
    out_dir: str = "pca_permod_umap",
    whiten: bool = False,
    seed: int = 0,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    scatter_png: str = "umap_2d.png",
) -> str:
    """
    Per-modality IncrementalPCA, concatenate embeddings, then UMAP to 2D.
    """
    assert X.ndim == 5 and X.shape[1:] == (3, 5, 20, 20), "X must have shape (N,3,5,20,20)"
    os.makedirs(out_dir, exist_ok=True)

    N = X.shape[0]
    K = 3 * ncomp_per_mod

    # Initialize scalers and PCA per modality
    scalers = [StandardScaler() for _ in range(3)]
    ipcas = [IncrementalPCA(n_components=ncomp_per_mod, batch_size=batch_size) for _ in range(3)]

    # Pass 1: fit scalers
    print("Fitting scalers...")
    for x0, x1, x2 in _iter_batches_permod(X, batch_size):
        scalers[0].partial_fit(x0)
        scalers[1].partial_fit(x1)
        scalers[2].partial_fit(x2)

    # Pass 2: fit IncrementalPCA per modality
    print("Fitting PCA models...")
    for x0, x1, x2 in _iter_batches_permod(X, batch_size):
        x0 = scalers[0].transform(x0)
        x1 = scalers[1].transform(x1)
        x2 = scalers[2].transform(x2)
        ipcas[0].partial_fit(x0)
        ipcas[1].partial_fit(x1)
        ipcas[2].partial_fit(x2)

    # Pass 3: transform and concatenate
    print("Transforming data...")
    emb_path = os.path.join(out_dir, "embeddings_permod.npy")
    Z = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, K))

    i = 0
    for x0, x1, x2 in _iter_batches_permod(X, batch_size):
        # Scale
        x0 = scalers[0].transform(x0)
        x1 = scalers[1].transform(x1)
        x2 = scalers[2].transform(x2)
        
        # PCA transform
        z0 = ipcas[0].transform(x0)
        z1 = ipcas[1].transform(x1)
        z2 = ipcas[2].transform(x2)
        
        # Optional whitening
        if whiten:
            z0 /= np.sqrt(np.maximum(ipcas[0].explained_variance_, 1e-12))
            z1 /= np.sqrt(np.maximum(ipcas[1].explained_variance_, 1e-12))
            z2 /= np.sqrt(np.maximum(ipcas[2].explained_variance_, 1e-12))
        
        # Concatenate and store
        Zb = np.concatenate([z0, z1, z2], axis=1).astype(np.float32)
        b = Zb.shape[0]
        Z[i:i+b] = Zb
        i += b

    # Save PCA artifacts
    print("Saving PCA artifacts...")
    np.save(os.path.join(out_dir, "explained_variance_ratio_permod.npy"),
            np.stack([ip.explained_variance_ratio_ for ip in ipcas], axis=0))
    np.save(os.path.join(out_dir, "components_permod.npy"),
            np.stack([ip.components_.astype(np.float32) for ip in ipcas], axis=0))
    
    for m in range(3):
        np.save(os.path.join(out_dir, f"feature_mean_mod{m}.npy"), scalers[m].mean_.astype(np.float32))
        np.save(os.path.join(out_dir, f"feature_scale_mod{m}.npy"), scalers[m].scale_.astype(np.float32))

    # UMAP on concatenated embeddings
    print("Running UMAP...")
    umap_model = UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=seed,
    )

    # Convert memmap to array for UMAP
    Z_array = np.array(Z)
    Y = umap_model.fit_transform(Z_array)
    
    # Save results
    np.save(os.path.join(out_dir, "umap_2d.npy"), Y.astype(np.float32))
    save_scatter(os.path.join(out_dir, scatter_png), Y)
    
    print(f"Pipeline complete. Results saved to: {out_dir}")
    return out_dir


def save_scatter(p: str, Y: np.ndarray):
    """Save a simple scatter plot of UMAP results."""
    plt.figure(figsize=(7, 6), dpi=120)
    
    # Default case: no coloring
    colours = None
    show_colorbar = False
    
    # to colour by session
    # if os.path.exists(os.path.join(os.path.dirname(p), "all_sessions_info.npy")):
    #     info = np.load(os.path.join(os.path.dirname(p), "all_sessions_info.npy"), allow_pickle=True).item()
    #     cell_counts = info.get('session_cell_counts', [])
    #     names = info.get('session_names', [])
    #     if cell_counts and names:
    #         colours = np.concatenate([np.ones(c) * i for i, c in enumerate(cell_counts)])

    # Color by cluster labels
    if os.path.exists(os.path.join(os.path.dirname(p), "umap_cluster_labels.npy")):
        cluster_labels = np.load(os.path.join(os.path.dirname(p), "umap_cluster_labels.npy"))
        # if clustering within cluster 7
        # Y = Y[cluster_labels == 7]
        # cluster_labels = np.load(os.path.join(os.path.dirname(p), "umap_cluster_labels_within7.npy"))
        colours = cluster_labels
        show_colorbar = True

    scatter = plt.scatter(Y[:, 0], Y[:, 1], s=0.1, alpha=0.1, c=colours, cmap='tab10')
    
    # Add colorbar for cluster labels
    if show_colorbar and colours is not None:
        unique_labels = np.unique(colours)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise points (-1)
        plt.colorbar(scatter, label=f'Cluster ID ({n_clusters} clusters)', shrink=0.8)
    
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP 2D Embedding")
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()


def clustering(OUT_DIR):

    Y = np.load(os.path.join(OUT_DIR, "umap_2d.npy"))
    # original_labels = np.load(os.path.join(OUT_DIR, "umap_cluster_labels.npy"))
    # Y = Y[original_labels == 7]
    # Perform clustering on Y
    clusterer = HDBSCAN(min_cluster_size=50)
    cluster_labels = clusterer.fit_predict(Y)
    # Save cluster labels
    np.save(os.path.join(OUT_DIR, "umap_cluster_labels.npy"), cluster_labels)


def cluster_representatives(umap_2d, labels, full_features, save_path=None, save_format='json'):
    """
    For each cluster, compute:
    - mean: mean of full_features for cluster
    - median: real datapoint closest to cluster centroid in UMAP space
    Returns a dict with keys 'mean' and 'median', each a list of arrays (len=n_clusters).
    Optionally saves the result to file (json or npy).
    """
    import numpy as np
    unique_labels = np.unique(labels)
    means = []
    medians = []
    for cl in unique_labels:
        idx = np.where(labels == cl)[0]
        umap_cluster = umap_2d[idx]
        features_cluster = full_features[idx]
        # Mean: mean of full_features in cluster
        mean_feat = features_cluster.mean(axis=0)
        means.append(mean_feat.tolist())
        # Median: find point closest to centroid in UMAP space
        centroid = umap_cluster.mean(axis=0)
        dists = np.linalg.norm(umap_cluster - centroid, axis=1)
        median_idx = idx[np.argmin(dists)]
        median_feat = full_features[median_idx]
        medians.append(median_feat.tolist())
    result = {'mean': means, 'median': medians}
    if save_path:
        if save_format == 'json':
            with open(save_path, 'w') as f:
                json.dump(result, f)
        elif save_format == 'npy':
            np.save(save_path, result)
    return result



if __name__ == "__main__":
    INPUT_PATH = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output\all_sessions_patches.npy"
    OUT_DIR = r"\\znas.cortexlab.net\Lab\Share\Ali\for-suyash\output"
    
    # print(f"Loading data from: {INPUT_PATH}")
    # X = np.load(INPUT_PATH, mmap_mode='r')
    # print(f"Data shape: {X.shape}")
    
    # run_permod_pca_umap(
    #     X=X,
    #     ncomp_per_mod=32,
    #     batch_size=4096,
    #     out_dir=OUT_DIR,
    #     whiten=False,
    #     seed=0,
    #     umap_neighbors=30,
    #     umap_min_dist=0.1,
    #     scatter_png="umap_2d.png",
    # )

    Y = np.load(os.path.join(OUT_DIR, "umap_2d.npy"))
    labels = np.load(os.path.join(OUT_DIR, "umap_cluster_labels.npy"))
    full_features = np.load(INPUT_PATH)
    cluster_reps = cluster_representatives(Y, labels, full_features, save_path=os.path.join(OUT_DIR, "cluster_representatives.npy"), save_format='npy')
    # save_scatter(os.path.join(OUT_DIR, "umap_2d_within7.png"), Y.astype(np.float32))

    # clustering(OUT_DIR)
