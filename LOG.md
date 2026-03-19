# Reproduction Log

## Day 1 (Thursday) — Setup & Data Exploration
- Downloaded 4 datasets from Figshare
- Set up GitHub repo with organized structure
- Explored all datasets:
  - DB1 Weather: Daily PRISM data, 5 sites, 1981–2023, columns: Site, date, rain, tmin, tmean, tmax, tdmean, vpdmin, vpdmax
  - DB2 Phenology: Site, Block_ID, Season, Budbreak, Flowering, Veraison, Harvest
  - DB3 Yield: Site, Block_ID, Season, Avg_Harvest_DOY, Yield_tha
  - DB4 Fruit composition: 15 columns including 12 analytes (2017–2023)
- Note: Weather data has no "Season" column — must extract year from date
- Created reusable src/ modules for clean code

## Day 2 (Friday) — Feature Engineering & Clustering
- [ ] Extract heat features for all site×year combinations
- [ ] Run HCA, optimize k, select 3 clusters
- [ ] PCA visualization
- [ ] RF validation (50 bootstrap iterations)
- [ ] Identify cluster ↔ season type mapping

## Day 3 (Saturday) — Yield & Quality Analysis
- [ ] Link clusters to yield and harvest date
- [ ] Run linear mixed models
- [ ] Fruit composition analysis (12 analytes)
- [ ] Generate final figures
- [ ] Write up summary for meeting
