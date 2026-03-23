import pandas as pd
import numpy as np

def identify_heatwaves(tmax_values, threshold=38.0):
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
                heatwaves.append({'start': streak_start, 'end': i-1, 'duration': duration,
                    'hwu': float(np.sum(hw_temps - threshold)), 'max_tmax': float(hw_temps.max())})
            streak_start = None
    if streak_start is not None:
        duration = len(above) - streak_start
        if duration >= 2:
            hw_temps = tmax[streak_start:]
            heatwaves.append({'start': streak_start, 'end': len(above)-1, 'duration': duration,
                'hwu': float(np.sum(hw_temps - threshold)), 'max_tmax': float(hw_temps.max())})
    return heatwaves

def validate_hwu_formula():
    hws = identify_heatwaves([36, 40, 42, 39, 35])
    assert len(hws)==1 and hws[0]['duration']==3 and abs(hws[0]['hwu']-7)<0.01
    print(f'✅ HWU: [40,42,39] → HWU=7°C (2+4+1)')
    assert len(identify_heatwaves([39, 40, 35]))==1
    print(f'✅ Minimum heatwave (2 days) detected')
    assert len(identify_heatwaves([35, 40, 35]))==0
    print(f'✅ Single hot day excluded')
    assert len(identify_heatwaves([39,40,35,39,41,38,35]))==2
    print(f'✅ Two separate heatwaves detected')
    print(f'\n✅ All validation tests passed!')

def compute_heat_features(tmax_values, vpd_max_values=None, tavg_values=None, rain_values=None, prefix=''):
    tmax = np.array(tmax_values, dtype=float)
    f = {}
    p = f'{prefix}_' if prefix else ''
    f[f'{p}tmax_max'] = float(tmax.max()) if len(tmax)>0 else np.nan
    f[f'{p}tmax_mean'] = float(tmax.mean()) if len(tmax)>0 else np.nan
    if tavg_values is not None: f[f'{p}tavg_mean'] = float(np.mean(tavg_values))
    if rain_values is not None: f[f'{p}precip_total'] = float(np.sum(rain_values))
    f[f'{p}heat_days'] = int(np.sum(tmax >= 38))
    hws = identify_heatwaves(tmax)
    f[f'{p}no_hw'] = len(hws)
    f[f'{p}tot_hwu'] = sum(h['hwu'] for h in hws)
    f[f'{p}avg_hwu'] = float(np.mean([h['hwu'] for h in hws])) if hws else 0.0
    f[f'{p}max_hw_dur'] = max(h['duration'] for h in hws) if hws else 0
    f[f'{p}avg_hw_dur'] = float(np.mean([h['duration'] for h in hws])) if hws else 0.0
    if vpd_max_values is not None and hws:
        vpd = np.array(vpd_max_values, dtype=float)
        hw_vpds = [float(vpd[h['start']:h['end']+1].max()) for h in hws if h['end']<len(vpd)]
        f[f'{p}max_hw_vpd'] = max(hw_vpds) if hw_vpds else 0.0
        f[f'{p}avg_hw_vpd'] = float(np.mean(hw_vpds)) if hw_vpds else 0.0
    else:
        f[f'{p}max_hw_vpd'] = 0.0
        f[f'{p}avg_hw_vpd'] = 0.0
    return f

def build_feature_matrix(weather_df, phenology_df=None):
    df = weather_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['_year'] = df['date'].dt.year
    df['_month'] = df['date'].dt.month
    df['_doy'] = df['date'].dt.dayofyear
    pheno_lookup = None
    if phenology_df is not None:
        pheno = phenology_df.copy()
        # Convert date strings to DOY (day of year) integers
        for col in ['Budbreak', 'Flowering', 'Veraison', 'Harvest']:
            if col in pheno.columns:
                pheno[col] = pd.to_datetime(pheno[col], errors='coerce').dt.dayofyear
        # Average DOY across blocks within each site×year
        pheno_lookup = pheno.groupby(['Site','Season']).agg(
            {'Budbreak':'mean','Flowering':'mean','Veraison':'mean','Harvest':'mean'}).reset_index()
        for col in ['Budbreak','Flowering','Veraison','Harvest']:
            pheno_lookup[col] = pheno_lookup[col].round().astype('Int64')
        print(f'Phenology: {pheno_lookup.dropna().shape[0]} site×year with complete records')
    all_features = []
    sites = sorted(df['Site'].unique())
    years = sorted(df['_year'].unique())
    print(f'Processing {len(sites)} sites × {len(years)} years ...')
    for site in sites:
        for year in years:
            sy = df[(df['Site']==site) & (df['_year']==year)]
            gs = sy[(sy['_month']>=4) & (sy['_month']<=10)].copy()
            if len(gs)<30: continue
            feats = {'site': site, 'year': int(year)}
            tmax = gs['tmax'].values
            tavg = gs['tmean'].values
            vpd = gs['vpdmax'].values if 'vpdmax' in gs.columns else None
            rain = gs['rain'].values if 'rain' in gs.columns else None
            