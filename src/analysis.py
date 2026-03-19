"""
Link weather clusters to yield, harvest date, and fruit composition.

Statistical approach:
- Linear mixed model (fixed: cluster, random: vineyard block)
- Fallback: Kruskal-Wallis + pairwise Wilcoxon
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


# ── Color scheme matching the paper ──
ORDER = ['POST-V', 'PRE-V', 'Cool']
PALETTE = {'POST-V': '#E74C3C', 'PRE-V': '#F39C12', 'Cool': '#27AE60'}


def merge_clusters(data_df, feature_matrix, site_col='Site', year_col='Season'):
    """
    Merge cluster/season_type labels onto a data DataFrame.
    
    Parameters
    ----------
    data_df : DataFrame
        Yield or fruit composition data with Site and Season columns.
    feature_matrix : DataFrame
        Must have 'site', 'year', and 'season_type' columns.
    site_col, year_col : str
        Column names in data_df for site and year.
    
    Returns
    -------
    DataFrame
        Merged data with 'season_type' column added.
    """
    clusters = feature_matrix[['site', 'year', 'season_type']].rename(
        columns={'site': site_col, 'year': year_col}
    )
    merged = data_df.merge(clusters, on=[site_col, year_col], how='inner')
    print(f'Merged: {len(merged)} rows')
    print(f'Season types: {merged["season_type"].value_counts().to_dict()}')
    return merged


def filter_blocks(merged, block_col='Block_ID', min_obs=2, min_clusters=2):
    """
    Keep only vineyard blocks with ≥min_obs in ≥min_clusters season types.
    This ensures fair paired comparisons.
    """
    bc = merged.groupby([block_col, 'season_type']).size().unstack(fill_value=0)
    valid = bc[(bc >= min_obs).sum(axis=1) >= min_clusters].index
    filtered = merged[merged[block_col].isin(valid)]
    print(f'After filter: {len(filtered)} rows, {filtered[block_col].nunique()} blocks')
    return filtered


def run_mixed_model(data, response_col, block_col='Block_ID'):
    """
    Fit linear mixed model: response ~ season_type, random=block.
    Falls back to Kruskal-Wallis if mixed model fails.
    """
    try:
        md = data[[response_col, 'season_type', block_col]].dropna()
        formula = f'Q("{response_col}") ~ C(season_type, Treatment(reference="Cool"))'
        model = smf.mixedlm(formula, data=md, groups=md[block_col]).fit()
        print(f'\nMixed Model for {response_col}:')
        print(model.summary().tables[1])
        return model
    except Exception as e:
        print(f'\nMixed model failed ({e}), using Kruskal-Wallis:')
        groups = [data[data['season_type'] == t][response_col].dropna()
                  for t in ORDER if t in data['season_type'].values]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            print(f'  H = {stat:.2f}, p = {p:.6f}')
        return None


def analyze_yield_harvest(yield_df, feature_matrix):
    """
    Full yield + harvest date analysis. Reproduces Figure 6.
    
    Parameters
    ----------
    yield_df : DataFrame
        Columns: Site, Block_ID, Season, Avg_Harvest_DOY, Yield_tha
    feature_matrix : DataFrame
        With 'site', 'year', 'season_type' columns.
    
    Returns
    -------
    DataFrame
        Merged and filtered data.
    """
    merged = merge_clusters(yield_df, feature_matrix)
    merged = filter_blocks(merged)

    metrics = [
        ('Avg_Harvest_DOY', 'Harvest Date', 'DOY'),
        ('Yield_tha', 'Yield', 'tons/ha'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (col, title, unit) in enumerate(metrics):
        if col not in merged.columns:
            print(f'\n⚠️  Column "{col}" not found, skipping.')
            continue

        print(f'\n{"="*55}')
        print(f'{title} ({unit}) by Season Type')
        print(f'{"="*55}')

        summary = merged.groupby('season_type')[col].agg(['mean', 'std', 'count'])
        print(summary.round(2))

        # Mixed model
        run_mixed_model(merged, col)

        # Change vs Cool
        if 'Cool' in summary.index:
            cool_val = summary.loc['Cool', 'mean']
            print(f'\nChanges vs Cool ({cool_val:.1f} {unit}):')
            for st in ['POST-V', 'PRE-V']:
                if st in summary.index:
                    diff = summary.loc[st, 'mean'] - cool_val
                    pct = (diff / cool_val) * 100
                    print(f'  {st}: {diff:+.1f} {unit} ({pct:+.1f}%)')

        # ── Plot: distribution ──
        ax1 = axes[row, 0]
        for st in ORDER:
            d = merged[merged['season_type'] == st][col].dropna()
            if len(d) > 0:
                ax1.hist(d, bins=25, alpha=0.45, color=PALETTE[st],
                         label=f'{st} (μ={d.mean():.1f})', density=True)
        ax1.set(xlabel=f'{title} ({unit})', ylabel='Density',
                title=f'{title} — Distribution')
        ax1.legend()

        # ── Plot: change vs Cool ──
        ax2 = axes[row, 1]
        if 'Cool' in summary.index:
            cool_val = summary.loc['Cool', 'mean']
            for i, st in enumerate(['POST-V', 'PRE-V']):
                if st in summary.index:
                    diff = summary.loc[st, 'mean'] - cool_val
                    if col == 'Yield_tha':
                        val = (diff / cool_val) * 100
                        lbl = f'{val:.0f}%'
                        ylabel = 'Yield Change vs Cool (%)'
                    else:
                        val = diff
                        lbl = f'{val:.0f} days'
                        ylabel = 'Change vs Cool (days)'
                    ax2.bar(i, val, color=PALETTE[st], width=0.5)
                    offset = -abs(val) * 0.15
                    ax2.text(i, val + offset, lbl, ha='center',
                             fontweight='bold', fontsize=12)
            ax2.set_xticks([0, 1])
            ax2.set_xticklabels(['POST-V', 'PRE-V'])
            ax2.set(ylabel=ylabel, title=f'{title} — Change vs Cool')
            ax2.axhline(0, color='red', ls='--', alpha=0.5)

    plt.suptitle('Figure 6 — Effect of Heat Extremes on Harvest Date & Yield',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig6_yield_harvest.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('\nSaved: figures/fig6_yield_harvest.png')

    return merged


def analyze_fruit_composition(fruit_df, feature_matrix):
    """
    Fruit composition analysis across season types. Reproduces Figure 7.
    
    Parameters
    ----------
    fruit_df : DataFrame
        Columns: Site, Block_ID, Season, + analyte columns
    feature_matrix : DataFrame
        With 'site', 'year', 'season_type' columns.
    
    Returns
    -------
    DataFrame
        Summary of p-values and means by season type.
    """
    # Known analyte columns from the paper
    analyte_cols = [
        '1-Octen-3-ol', 'C6 compounds', 'IBMP', 'β-Damascenone',
        'Total anthocyanins', 'Polymeric tannins', 'Quercetin glycosides',
        'TSS', 'Moisture', 'pH', 'Malic acid', 'YAN'
    ]

    merged = merge_clusters(fruit_df, feature_matrix)
    available = [a for a in analyte_cols if a in merged.columns]
    print(f'{len(available)} of {len(analyte_cols)} analytes found\n')

    results = []
    for analyte in available:
        groups = {}
        for t in ORDER:
            d = merged[merged['season_type'] == t][analyte].dropna()
            if len(d) > 0:
                groups[t] = d
        if len(groups) < 2:
            continue

        stat, p = kruskal(*groups.values())
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        row = {'analyte': analyte, 'p': p, 'sig': sig}
        for t, d in groups.items():
            row[f'{t}_mean'] = d.mean()
        results.append(row)

        means_str = '  '.join(f'{t}={d.mean():.1f}' for t, d in groups.items())
        print(f'{analyte:25s}  p={p:.4f} {sig:3s}  |  {means_str}')

    # ── Grid plot ──
    n = len(available)
    ncols = 4
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for i, analyte in enumerate(available):
        d = merged[merged['season_type'].isin(ORDER)].dropna(subset=[analyte])
        sns.boxplot(data=d, x='season_type', y=analyte, order=ORDER,
                    palette=PALETTE, ax=axes_flat[i], width=0.6)
        sig_str = next((r['sig'] for r in results if r['analyte'] == analyte), '')
        axes_flat[i].set_title(f'{analyte} ({sig_str})', fontsize=10)
        axes_flat[i].set_xlabel('')

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle('Figure 7 — Fruit Composition by Season Type',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('figures/fig7_fruit_composition.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('\nSaved: figures/fig7_fruit_composition.png')

    return pd.DataFrame(results)
