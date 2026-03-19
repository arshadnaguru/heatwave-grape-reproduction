"""
Hierarchical Clustering Analysis (HCA) of growing seasons.

Groups site×year combinations into 3 season types based on heat patterns:
- Cool (C3): No significant heat events
- Preveraison heat (C2/PRE-V): Heatwaves before veraison (Jun–Jul)
- Postveraison heat (C1/POST-V): Heatwaves after veraison (Aug–Sep)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def run_hca(feature_matrix, n_clusters=3):
    """
    Run Hierarchical Clustering Analysis with Ward's method.
    
    Parameters
    ----------
    feature_matrix : DataFrame
        Output of build_feature_matrix(). Must have 'site' and 'year' columns.
    n_clusters : int
        Number of clusters to cut (default 3, as in the paper).
    
    Returns
    -------
    labels : ndarray
        Cluster label (1, 2, or 3) for each row.
    Z : ndarray
        Linkage matrix.
    X_scaled : ndarray
        Standardized feature values.
    feat_cols : list
        Names of feature columns used.
    """
    feat_cols = [c for c in feature_matrix.columns if c not in ['site', 'year']]
    X = feature_matrix[feat_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(X_scaled, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    print('Cluster sizes:')
    for c in sorted(np.unique(labels)):
        print(f'  Cluster {c}: n = {(labels == c).sum()}')
    print(f'Paper expects: ~21, ~70, ~124')

    return labels, Z, X_scaled, feat_cols


def plot_optimization(X_scaled):
    """
    Plot WSS (elbow) and silhouette score for k = 2..10.
    Reproduces Supplemental Figure 6.
    """
    ks = range(2, 11)
    wss_list, sil_list = [], []

    for k in ks:
        Z = linkage(X_scaled, method='ward')
        labs = fcluster(Z, t=k, criterion='maxclust')
        wss = sum(
            np.sum((X_scaled[labs == c] - X_scaled[labs == c].mean(axis=0)) ** 2)
            for c in np.unique(labs)
        )
        wss_list.append(wss)
        sil_list.append(silhouette_score(X_scaled, labs))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    a1.plot(list(ks), wss_list, 'bo-', lw=2)
    a1.axvline(3, color='red', ls='--', alpha=0.7, label='k = 3')
    a1.set(xlabel='Number of clusters (k)', ylabel='Within-cluster Sum of Squares',
           title='A) Elbow Method')
    a1.legend()

    a2.plot(list(ks), sil_list, 'go-', lw=2)
    a2.axvline(3, color='red', ls='--', alpha=0.7, label='k = 3')
    a2.set(xlabel='Number of clusters (k)', ylabel='Silhouette Score',
           title='B) Silhouette Method')
    a2.legend()

    plt.tight_layout()
    plt.savefig('figures/cluster_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: figures/cluster_optimization.png')


def plot_pca(X_scaled, labels):
    """
    PCA score plot colored by cluster. Reproduces Figure 3A.
    """
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {1: '#E74C3C', 2: '#F39C12', 3: '#27AE60'}
    markers = {1: 'o', 2: '^', 3: 's'}

    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(scores[m, 0], scores[m, 1],
                   c=colors.get(c, 'gray'), marker=markers.get(c, 'o'),
                   label=f'Cluster {c} (n={m.sum()})',
                   alpha=0.65, s=60, edgecolors='white', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('HCA Clusters — PCA Score Plot (Figure 3A)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/fig3_pca_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: figures/fig3_pca_clusters.png')


def identify_cluster_types(feature_matrix):
    """
    Print heat stats by cluster to identify Cool / PRE-V / POST-V.
    
    HCA cluster numbers are arbitrary — this function helps you
    map them to the paper's season types.
    """
    print('=== Cluster Identification ===\n')
    for c in sorted(feature_matrix['cluster'].unique()):
        sub = feature_matrix[feature_matrix['cluster'] == c]
        print(f'Cluster {c} (n={len(sub)}):')
        for col in ['season_heat_days', 'season_no_hw', 'season_tmax_max',
                     'MayJul_heat_days', 'AugOct_heat_days',
                     'MayJul_no_hw', 'AugOct_no_hw']:
            if col in sub.columns:
                print(f'  {col:25s}: {sub[col].mean():.2f}')
        print()

    print('MAPPING GUIDE:')
    print('  Cool    → ~0 heat days, ~0 heatwaves      (LARGEST group, ~124)')
    print('  PRE-V   → heat in MayJul > AugOct          (MEDIUM group, ~70)')
    print('  POST-V  → heat in AugOct > MayJul, highest overall (SMALLEST, ~21)')
