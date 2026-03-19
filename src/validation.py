"""
Cluster validation using bootstrap Random Forest classification.

Method: 50 iterations of 50/50 train-test split with 500 trees.
Expected result: ~1.35% average test error rate (TER).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def validate_rf(X_scaled, labels, feat_cols, n_iter=50, n_trees=500, max_features=7):
    """
    Bootstrap Random Forest validation of clusters.
    Reproduces Figure 4 (TER plot + variable importance).
    
    Parameters
    ----------
    X_scaled : ndarray
        Standardized feature matrix.
    labels : ndarray
        Cluster labels.
    feat_cols : list
        Feature column names (for importance labeling).
    n_iter : int
        Number of bootstrap iterations (default 50).
    n_trees : int
        Trees per forest (default 500).
    max_features : int
        Random features per tree (default 7).
    
    Returns
    -------
    ters : list
        Test error rate (%) per iteration.
    imp_df : DataFrame
        Feature importance ranking.
    """
    ters = []
    imps = []

    for i in range(n_iter):
        Xtr, Xte, ytr, yte = train_test_split(
            X_scaled, labels, test_size=0.5, random_state=i
        )
        rf = RandomForestClassifier(
            n_estimators=n_trees, max_features=max_features, random_state=i
        )
        rf.fit(Xtr, ytr)
        ter = (1 - accuracy_score(yte, rf.predict(Xte))) * 100
        ters.append(ter)
        imps.append(rf.feature_importances_)

    avg_ter = np.mean(ters)
    avg_imp = np.mean(imps, axis=0)

    # ── Plot ──
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))

    # A) TER per iteration
    a1.scatter(range(1, n_iter+1), ters, c='green', s=30, alpha=0.7)
    a1.axhline(avg_ter, color='red', ls='--', lw=1.5,
               label=f'Avg TER: {avg_ter:.2f}%')
    a1.set(xlabel='Iteration', ylabel='TER (%)',
           title='A) Test Error Rate (Figure 4A)')
    a1.legend(fontsize=11)
    a1.set_ylim(bottom=-0.5)

    # B) Variable importance (top 20)
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': avg_imp})
    imp_df = imp_df.sort_values('importance', ascending=True)
    top = imp_df.tail(20)
    threshold = imp_df['importance'].quantile(0.85)
    colors = ['#E67E22' if v > threshold else '#95A5A6' for v in top['importance']]

    a2.barh(range(len(top)), top['importance'].values, color=colors)
    a2.set_yticks(range(len(top)))
    a2.set_yticklabels(top['feature'].values, fontsize=8)
    a2.set(xlabel='Mean Decrease in Accuracy',
           title='B) Variable Importance — Top 20 (Figure 4B)')

    plt.tight_layout()
    plt.savefig('figures/fig4_rf_validation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\nAverage TER: {avg_ter:.2f}%  (paper reports 1.35%)')
    print(f'\nTop 10 most important features:')
    print(imp_df.tail(10).iloc[::-1][['feature', 'importance']].to_string(index=False))
    print(f'\nSaved: figures/fig4_rf_validation.png')

    return ters, imp_df
