"""
Hierarchical Clustering Analysis (HCA) with:
- Auto-labeling clusters (not hardcoded)
- Heat distribution plots per cluster (temporal validation)
- Full optimization suite
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def run_hca(feature_matrix, n_clusters=3):
    """Run HCA with Ward's method."""
    feat_cols = [c for c in feature_matrix.columns if c not in ['site', 'year']]
    X = feature_matrix[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Z = linkage(X_scaled, method='ward')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    print('Cluster sizes:')
    for c in sorted(np.unique(labels)):
        print(f'  Cluster {c}: n = {(labels == c).sum()}')
    return labels, Z, X_scaled, feat_cols


def auto_label_clusters(feature_matrix):
    """
    Automatically label clusters as Cool / PRE-V / POST-V
    based on heat feature patterns. No hardcoding.

    Logic (from paper's cluster characteristics, Figure 5):
    - Cool: lowest total heat days
    - PRE-V: among hot clusters, higher MayJul heat than AugOct
    - POST-V: among hot clusters, higher AugOct heat than MayJul
    """
    summary = feature_matrix.groupby('cluster').agg({
        'season_heat_days': 'mean',
        'MayJul_heat_days': 'mean',
        'AugOct_heat_days': 'mean',
        'season_no_hw': 'mean',
    }).round(2)

    print('=== Auto-labeling clusters ===\n')
    print(summary.to_string())

    # Step 1: Cool = cluster with lowest heat days
    cool_cluster = summary['season_heat_days'].idxmin()

    # Step 2: Among remaining, PRE-V has higher MayJul, POST-V has higher AugOct
    hot_clusters = [c for c in summary.index if c != cool_cluster]
    if len(hot_clusters) == 2:
        c_a, c_b = hot_clusters
        if summary.loc[c_a, 'MayJul_heat_days'] > summary.loc[c_b, 'MayJul_heat_days']:
            prev_cluster, postv_cluster = c_a, c_b
        else:
            prev_cluster, postv_cluster = c_b, c_a
    else:
        prev_cluster = hot_clusters[0]
        postv_cluster = hot_clusters[0]

    cluster_map = {
        cool_cluster: 'Cool',
        prev_cluster: 'PRE-V',
        postv_cluster: 'POST-V',
    }

    feature_matrix['season_type'] = feature_matrix['cluster'].map(cluster_map)

    print(f'\nAuto-mapping:')
    for c, label in sorted(cluster_map.items()):
        n = (feature_matrix['cluster'] == c).sum()
        hd = summary.loc[c, 'season_heat_days']
        mj = summary.loc[c, 'MayJul_heat_days']
        ao = summary.loc[c, 'AugOct_heat_days']
        print(f'  Cluster {c} → {label:7s} (n={n:3d}, heat_days={hd:.1f}, MayJul={mj:.1f}, AugOct={ao:.1f})')

    print(f'\n{feature_matrix["season_type"].value_counts().to_string()}')
    return feature_matrix, cluster_map


def plot_optimization(X_scaled):
    """WSS + Silhouette for k=2..10. Reproduces Supplemental Figure 6."""
    ks = range(2, 11)
    wss, sil = [], []
    for k in ks:
        labs = fcluster(linkage(X_scaled, method='ward'), t=k, criterion='maxclust')
        w = sum(np.sum((X_scaled[labs==c] - X_scaled[labs==c].mean(0))**2) for c in np.unique(labs))
        wss.append(w)
        sil.append(silhouette_score(X_scaled, labs))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(list(ks), wss, 'bo-', lw=2)
    a1.axvline(3, color='red', ls='--', alpha=.7, label='k=3')
    a1.set(xlabel='k', ylabel='WSS', title='A) Elbow Method'); a1.legend()
    a2.plot(list(ks), sil, 'go-', lw=2)
    a2.axvline(3, color='red', ls='--', alpha=.7, label='k=3')
    a2.set(xlabel='k', ylabel='Silhouette', title='B) Silhouette'); a2.legend()
    plt.tight_layout()
    plt.savefig('figures/cluster_optimization.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca(X_scaled, labels):
    """PCA score plot. Reproduces Figure 3A."""
    pca = PCA(n_components=2)
    sc = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {1:'#E74C3C', 2:'#F39C12', 3:'#27AE60'}
    markers = {1:'o', 2:'^', 3:'s'}
    for c in sorted(np.unique(labels)):
        m = labels == c
        ax.scatter(sc[m,0], sc[m,1], c=colors.get(c,'gray'), marker=markers.get(c,'o'),
                   label=f'Cluster {c} (n={m.sum()})', alpha=.65, s=60,
                   edgecolors='white', linewidth=.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('HCA Clusters — PCA Score Plot (Figure 3A)')
    ax.legend(); ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig('figures/fig3_pca_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_heat_distribution(feature_matrix):
    """
    Temporal validation: show heat distribution by month per cluster.
    Reproduces Figure 5B/5C — confirms PRE-V peaks May-Jul, POST-V peaks Aug-Sep.
    """
    palette = {'POST-V':'#E74C3C', 'PRE-V':'#F39C12', 'Cool':'#27AE60'}
    months = ['Jun', 'Jul', 'Aug', 'Sep']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # A) Heat days by month
    data = []
    for st in ['POST-V', 'PRE-V', 'Cool']:
        sub = feature_matrix[feature_matrix['season_type'] == st]
        for m in months:
            col = f'{m}_heat_days'
            if col in sub.columns:
                data.append({'season_type': st, 'month': m, 'heat_days': sub[col].mean()})
    ddf = pd.DataFrame(data)
    for st in ['POST-V', 'PRE-V', 'Cool']:
        sub = ddf[ddf['season_type'] == st]
        axes[0].plot(sub['month'], sub['heat_days'], 'o-', color=palette[st],
                     label=st, lw=2, markersize=8)
    axes[0].set(xlabel='Month', ylabel='Avg heat days (Tmax ≥ 38°C)',
                title='A) Heat Days by Month (Fig 5B)')
    axes[0].legend()

    # B) Heatwaves by month
    data2 = []
    for st in ['POST-V', 'PRE-V', 'Cool']:
        sub = feature_matrix[feature_matrix['season_type'] == st]
        for m in months:
            col = f'{m}_no_hw'
            if col in sub.columns:
                data2.append({'season_type': st, 'month': m, 'heatwaves': sub[col].mean()})
    ddf2 = pd.DataFrame(data2)
    for st in ['POST-V', 'PRE-V', 'Cool']:
        sub = ddf2[ddf2['season_type'] == st]
        axes[1].plot(sub['month'], sub['heatwaves'], 'o-', color=palette[st],
                     label=st, lw=2, markersize=8)
    axes[1].set(xlabel='Month', ylabel='Avg heatwaves',
                title='B) Heatwaves by Month (Fig 5C)')
    axes[1].legend()

    # C) Tmax by month
    data3 = []
    for st in ['POST-V', 'PRE-V', 'Cool']:
        sub = feature_matrix[feature_matrix['season_type'] == st]
        for m in months:
            col = f'{m}_tmax_max'
            if col in sub.columns:
                data3.append({'season_type': st, 'month': m, 'tmax': sub[col].mean()})
    ddf3 = pd.DataFrame(data3)
    for st in ['POST-V', 'PRE-V', 'Cool']:
        sub = ddf3[ddf3['season_type'] == st]
        axes[2].plot(sub['month'], sub['tmax'], 'o-', color=palette[st],
                     label=st, lw=2, markersize=8)
    axes[2].axhline(38, color='black', ls='--', alpha=.5, label='38°C threshold')
    axes[2].set(xlabel='Month', ylabel='Max Tmax (°C)',
                title='C) Peak Temperature by Month (Fig 5A)')
    axes[2].legend()

    plt.suptitle('Temporal Validation: Heat Distribution by Cluster (Figure 5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/fig5_heat_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: figures/fig5_heat_distribution.png')

    # Print summary table
    print('\n=== Cluster Characteristics Summary ===')
    for st in ['Cool', 'PRE-V', 'POST-V']:
        sub = feature_matrix[feature_matrix['season_type'] == st]
        print(f'\n{st} (n={len(sub)}):')
        print(f'  Season Tmax max:     {sub["season_tmax_max"].mean():.1f}°C')
        print(f'  Season heat days:    {sub["season_heat_days"].mean():.1f}')
        print(f'  Season heatwaves:    {sub["season_no_hw"].mean():.1f}')
        print(f'  MayJul heat days:    {sub["MayJul_heat_days"].mean():.1f}')
        print(f'  AugOct heat days:    {sub["AugOct_heat_days"].mean():.1f}')
        if 'precip_season' in sub.columns or 'season_precip_total' in sub.columns:
            pcol = 'season_precip_total' if 'season_precip_total' in sub.columns else 'precip_season'
            if pcol in sub.columns:
                print(f'  Season precip:       {sub[pcol].mean():.0f} mm')
