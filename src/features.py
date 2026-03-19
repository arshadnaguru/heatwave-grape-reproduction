"""
Feature engineering: heatwave detection and weather summary features.

Heatwave definition (per paper): ≥2 consecutive days with Tmax ≥ 38°C
HWU (Heatwave Units): Σ(Tmax − 38) across all days of a heatwave
"""

import pandas as pd
import numpy as np


def identify_heatwaves(tmax_values, threshold=38.0):
    """
    Find heatwaves in a series of daily Tmax values.
    
    A heatwave = ≥2 consecutive days with Tmax ≥ threshold.
    
    Parameters
    ----------
    tmax_values : array-like
        Daily maximum temperatures.
    threshold : float
        Temperature threshold in °C (default 38°C ≈ 100°F).
    
    Returns
    -------
    list of dict
        Each dict has: start, end, duration, hwu, max_tmax.
    """
    tmax = np.array(tmax_values, dtype=float)
    above = tmax >= threshold
    heatwaves = []
    streak_start = None

    for i in range(len(above)):
        if above[i] and streak_start is None:
            streak_start = i
        elif not above[i] and streak_start is not None:
            duration = i - streak_start
            if duration >= 2:
                hw_temps = tmax[streak_start:i]
                heatwaves.append({
                    'start': streak_start,
                    'end': i - 1,
                    'duration': duration,
                    'hwu': float(np.sum(hw_temps - threshold)),
                    'max_tmax': float(hw_temps.max()),
                })
            streak_start = None

    # Handle streak running to end of array
    if streak_start is not None:
        duration = len(above) - streak_start
        if duration >= 2:
            hw_temps = tmax[streak_start:]
            heatwaves.append({
                'start': streak_start,
                'end': len(above) - 1,
                'duration': duration,
                'hwu': float(np.sum(hw_temps - threshold)),
                'max_tmax': float(hw_temps.max()),
            })

    return heatwaves


def compute_heat_features(tmax_values, vpd_max_values=None, prefix=''):
    """
    Compute heat-related summary features for a time period.
    
    Parameters
    ----------
    tmax_values : array-like
        Daily Tmax values for the period.
    vpd_max_values : array-like, optional
        Daily VPDmax values (same length as tmax_values).
    prefix : str
        Prefix for feature names (e.g., 'season', 'Jun', 'MayJul').
    
    Returns
    -------
    dict
        Feature name → value.
    """
    tmax = np.array(tmax_values, dtype=float)
    f = {}
    p = f'{prefix}_' if prefix else ''

    # Temperature stats
    f[f'{p}tmax_max'] = float(tmax.max()) if len(tmax) > 0 else np.nan
    f[f'{p}tmax_mean'] = float(tmax.mean()) if len(tmax) > 0 else np.nan

    # Heat days (Tmax ≥ 38°C)
    f[f'{p}heat_days'] = int(np.sum(tmax >= 38))

    # Heatwaves
    hws = identify_heatwaves(tmax)
    f[f'{p}no_hw'] = len(hws)
    f[f'{p}tot_hwu'] = sum(h['hwu'] for h in hws)
    f[f'{p}avg_hwu'] = float(np.mean([h['hwu'] for h in hws])) if hws else 0.0
    f[f'{p}max_hw_dur'] = max(h['duration'] for h in hws) if hws else 0
    f[f'{p}avg_hw_dur'] = float(np.mean([h['duration'] for h in hws])) if hws else 0.0

    # VPD during heatwaves
    if vpd_max_values is not None and hws:
        vpd = np.array(vpd_max_values, dtype=float)
        hw_vpds = []
        for h in hws:
            if h['end'] < len(vpd):
                hw_vpds.append(float(vpd[h['start']:h['end']+1].max()))
        f[f'{p}max_hw_vpd'] = max(hw_vpds) if hw_vpds else 0.0
        f[f'{p}avg_hw_vpd'] = float(np.mean(hw_vpds)) if hw_vpds else 0.0
    else:
        f[f'{p}max_hw_vpd'] = 0.0
        f[f'{p}avg_hw_vpd'] = 0.0

    return f


def build_feature_matrix(weather_df):
    """
    Extract heat features for every site × year combination.
    
    Parameters
    ----------
    weather_df : DataFrame
        Must have columns: Site, date, tmax, tmean, rain, vpdmax.
        The 'date' column is parsed to extract year and month.
    
    Returns
    -------
    DataFrame
        One row per site×year, with all heat features + site/year identifiers.
    """
    df = weather_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['_year'] = df['date'].dt.year
    df['_month'] = df['date'].dt.month

    all_features = []
    sites = sorted(df['Site'].unique())
    years = sorted(df['_year'].unique())

    print(f'Processing {len(sites)} sites × {len(years)} years ...')

    for site in sites:
        for year in years:
            sy = df[(df['Site'] == site) & (df['_year'] == year)]

            # Growing season: April–October
            gs = sy[(sy['_month'] >= 4) & (sy['_month'] <= 10)].copy()
            if len(gs) < 30:
                continue

            feats = {'site': site, 'year': int(year)}
            tmax = gs['tmax'].values
            tavg = gs['tmean'].values
            vpd = gs['vpdmax'].values if 'vpdmax' in gs.columns else None

            # ── Season-level (April–October) ──
            feats.update(compute_heat_features(tmax, vpd, prefix='season'))
            feats['gdd_apr_oct'] = float(np.sum(np.maximum(tavg - 10, 0)))
            if 'rain' in gs.columns:
                feats['precip_season'] = float(gs['rain'].sum())

            # ── Monthly features ──
            month_names = {4:'Apr', 5:'May', 6:'Jun', 7:'Jul',
                           8:'Aug', 9:'Sep', 10:'Oct'}
            for m, mname in month_names.items():
                md = gs[gs['_month'] == m]
                if len(md) == 0:
                    continue
                m_vpd = md['vpdmax'].values if vpd is not None else None
                feats.update(compute_heat_features(md['tmax'].values, m_vpd, prefix=mname))

            # ── Early vs Late season (pre/post veraison proxy) ──
            early = gs[(gs['_month'] >= 5) & (gs['_month'] <= 7)]
            late  = gs[(gs['_month'] >= 8) & (gs['_month'] <= 10)]

            if len(early) > 0:
                e_vpd = early['vpdmax'].values if vpd is not None else None
                feats.update(compute_heat_features(early['tmax'].values, e_vpd, prefix='MayJul'))
            if len(late) > 0:
                l_vpd = late['vpdmax'].values if vpd is not None else None
                feats.update(compute_heat_features(late['tmax'].values, l_vpd, prefix='AugOct'))

            all_features.append(feats)

    fm = pd.DataFrame(all_features)
    print(f'Done! {fm.shape[0]} rows × {fm.shape[1]} columns')
    print(f'(Paper expects ~213 rows)')
    return fm
