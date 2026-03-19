# Reproduction: Heatwave Impact on Cabernet Sauvignon

Reproduction of **Previtali et al. (2026)** — *"Long-term Weather Observations Reveal the Impact of Heatwaves on the Yield and Fruit Composition of Cabernet Sauvignon"*

- **Journal**: American Journal of Enology and Viticulture (AJEV), Vol. 77
- **DOI**: [10.5344/ajev.2025.25017](https://doi.org/10.5344/ajev.2025.25017)
- **Data**: [Figshare Repository](https://doi.org/10.6084/m9.figshare.28486232)

## What This Paper Does

1. Analyzes 43 years of daily weather data (1981–2023) across 5 California vineyard sites
2. Extracts heat-related features and classifies seasons into 3 types using **Hierarchical Clustering Analysis (HCA)**
3. Validates clusters with **Random Forest** (bootstrap, 50 iterations, ~1.35% error rate)
4. Links season types to grape **yield**, **harvest date**, and **fruit composition** to quantify heatwave damage

## Key Results

| Metric | Postveraison Heat (C1) | Preveraison Heat (C2) | Cool (C3) |
|--------|------------------------|----------------------|-----------|
| Harvest date change | -17 days | -13 days | reference |
| Yield change | -22% | -30% | reference |
| Yield (tons/ha) | 5.5 | 5.0 | 7.0 |

## Repository Structure

```
├── README.md
├── requirements.txt
├── LOG.md                     ← Daily progress log
├── data/                      ← Figshare datasets (not tracked by git)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering_validation.ipynb
│   └── 04_yield_quality_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── clustering.py
│   ├── validation.py
│   └── analysis.py
└── figures/
```

## How to Run (Google Colab)

1. Clone this repo in Colab:
```python
!git clone https://github.com/YOUR_USERNAME/heatwave-grape-reproduction.git
%cd heatwave-grape-reproduction
!pip install -r requirements.txt
```

2. Upload the 4 data files to `data/` (download from [Figshare](https://doi.org/10.6084/m9.figshare.28486232))

3. Run notebooks in order: `01` → `02` → `03` → `04`

## Author

Arshad Naguru — MS in Artificial Intelligence, Rochester Institute of Technology
