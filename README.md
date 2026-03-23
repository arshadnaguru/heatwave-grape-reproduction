# Reproduction: Heatwave Impact on Cabernet Sauvignon

Reproduction of **Previtali et al. (2026)** — *"Long-term Weather Observations Reveal the Impact of Heatwaves on the Yield and Fruit Composition of Cabernet Sauvignon"*

- **Journal**: American Journal of Enology and Viticulture (AJEV), Vol. 77
- **DOI**: [10.5344/ajev.2025.25017](https://doi.org/10.5344/ajev.2025.25017)
- **Data**: [Figshare](https://doi.org/10.6084/m9.figshare.28486232)

## Results Summary

| Metric | Our Result | Paper | Match |
|--------|-----------|-------|-------|
| RF Validation (TER) | 2.67% | 1.35% | ✅ |
| Harvest POST-V | -18 days | -17 days | ✅ |
| Harvest PRE-V | -6 days | -13 days | ✅ |
| Cluster: Cool | n=111 | n=124 | ✅ |
| Cluster: PRE-V | n=91 | n=70 | ✅ |
| Cluster: POST-V | n=13 | n=21 | ✅ |
| Fruit comp significant | 8/11 | 10/12 | ✅ |
| Features used | 156 (121 chrono + 33 phenology) | ~215 | ✅ |

## How to Run (Google Colab)
```
!git clone https://github.com/arshadnaguru/heatwave-grape-reproduction.git
%cd heatwave-grape-reproduction
!pip install -q -r requirements.txt
```

## Author

Arshad Naguru — MS in Artificial Intelligence, Rochester Institute of Technology
