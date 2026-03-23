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
    print('✅ HWU: [40,42,39] → HWU=7°C (2+4+1)')
    assert len(identify_heatwaves([39, 40, 35]))==1
    print('✅ Minimum heatwave (2 days) detected')
    assert len(identify_heatwaves([35, 40, 35]))==0
    print('✅ Single hot day excluded')
    assert len(identify_heatwaves([39,40,35,39,41,38,35]))==2
    print('✅ Two separate heatwaves detected')
    print('\n✅ All validation tests passed!')

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
        for col in ['Budbreak', 'Flowering', 'Veraison', 'Harvest']:
            if col in pheno.columns:
                pheno[col] = pd.to_datetime(pheno[col], errors='coerce').dt.dayofyear
        pheno_lookup = pheno.groupby(['Site','Season']).agg(
            {'Budbreak':'mean','Flowering':'mean','Veraison':'mean','Harvest':'mean'}).reset_index()
        for col in ['Budbreak','Flowering','Veraison','Harvest']:
            pheno_lookup[col] = pheno_lookup[col].round().astype('Int64')
        print(f'Phenology: {pheno_lookup.dropna().shape[0]} site x year with complete records')
    all_features = []
    sites = sorted(df['Site'].unique())
    years = sorted(df['_year'].unique())
    print(f'Processing {len(sites)} sites x {len(years)} years ...')
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
            feats.update(compute_heat_features(tmax, vpd, tavg, rain, prefix='season'))
            feats['gdd_apr_oct'] = float(np.sum(np.maximum(tavg-10, 0)))
            for m, mn in {4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct'}.items():
                md = gs[gs['_month']==m]
                if len(md)==0: continue
                feats.update(compute_heat_features(md['tmax'].values,
                    md['vpdmax'].values if vpd is not None else None,
                    md['tmean'].values, md['rain'].values if rain is not None else None, prefix=mn))
            for subset, pfx in [(gs[(gs['_month']>=5)&(gs['_month']<=7)],'MayJul'),
                                 (gs[(gs['_month']>=8)&(gs['_month']<=10)],'AugOct')]:
                if len(subset)>0:
                    feats.update(compute_heat_features(subset['tmax'].values,
                        subset['vpdmax'].values if vpd is not None else None,
                        subset['tmean'].values, subset['rain'].values if rain is not None else None, prefix=pfx))
            if pheno_lookup is not None:
                p_row = pheno_lookup[(pheno_lookup['Site']==site)&(pheno_lookup['Season']==year)]
                if len(p_row)>0:
                    pr = p_row.iloc[0]
                    fl = pr.get('Flowering')
                    ver = pr.get('Veraison')
                    har = pr.get('Harvest')
                    bb = pr.get('Budbreak')
                    if pd.notna(fl) and pd.notna(ver):
                        fl_ver = sy[(sy['_doy']>=int(fl))&(sy['_doy']<int(ver))]
                        if len(fl_ver)>0:
                            feats.update(compute_heat_features(fl_ver['tmax'].values,
                                fl_ver['vpdmax'].values if vpd is not None else None,
                                fl_ver['tmean'].values, prefix='FLtoVER'))
                    if pd.notna(ver) and pd.notna(har):
                        ver_h = sy[(sy['_doy']>=int(ver))&(sy['_doy']<=int(har))]
                        if len(ver_h)>0:
                            feats.update(compute_heat_features(ver_h['tmax'].values,
                                ver_h['vpdmax'].values if vpd is not None else None,
                                ver_h['tmean'].values, prefix='VERtoH'))
                    if pd.notna(bb) and pd.notna(har):
                        bb_h = sy[(sy['_doy']>=int(bb))&(sy['_doy']<=int(har))]
                        if len(bb_h)>0:
                            feats.update(compute_heat_features(bb_h['tmax'].values,
                                bb_h['vpdmax'].values if vpd is not None else None,
                                bb_h['tmean'].values, prefix='BBtoH'))
            all_features.append(feats)
    fm = pd.DataFrame(all_features)
    n_pheno = len([c for c in fm.columns if c.startswith(('FLtoVER','VERtoH','BBtoH'))])
    n_chrono = len([c for c in fm.columns if c not in ['site','year'] and not c.startswith(('FLtoVER','VERtoH','BBtoH'))])
    print(f'\nDone! {fm.shape[0]} rows x {fm.shape[1]} columns')
    print(f'  Chronological features: {n_chrono}')
    print(f'  Phenology-based features: {n_pheno}')
    return fm
