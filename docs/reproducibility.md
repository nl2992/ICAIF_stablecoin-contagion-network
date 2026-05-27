# Reproducibility

## Philosophy

The pipeline is fully reproducible given access to the same raw data sources. All
transformations from raw API responses to paper-ready tables are explicit, versioned,
and auditable.

## Steps to reproduce

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env  # fill in API keys

# 1. event window table
python scripts/00_make_event_windows.py
# → results/tables/table_event_windows.csv
# → results/tables/table_node_coverage.csv
# → results/figures/figure_heatmap_coverage.png

# 2. fetch raw data for one event
python scripts/01_fetch_raw_data.py --event usdc_svb_2023
# → data/raw/ (not committed; hashes in data/manifests/)

# 3. reconstruct books and pools
python scripts/02_build_books_and_pools.py --event usdc_svb_2023
# → data/silver/

# 4. build feature panel
python scripts/03_build_feature_panel.py --event usdc_svb_2023
# → data/gold/dataset_contagion_features_usdc_svb_2023.parquet

# 5–8. analysis (MVP)
make mvp EVENT=usdc_svb_2023

# 9. paper outputs
make paper
```

## Manifests

Every raw data fetch writes a manifest to `data/manifests/` with:
- Source URL or API endpoint
- Query parameters / time range
- SHA-256 hash of the raw response
- Timestamp of fetch
- Schema version

These manifests allow verification that results were not produced from modified or
manipulated data.

## Environment pinning

Pin exact package versions with:
```bash
pip freeze > requirements.lock
```

The `requirements.lock` file is committed alongside `requirements.txt`.

## Random seeds

All stochastic operations (bootstrap resampling, GNN training) use seeds specified in
`configs/models.yaml` under `random_seed`. Default seed: `42`.

## What is NOT reproducible without paid data

Some analyses require paid historical data (Tardis, Kaiko). These are noted in
`configs/sources.yaml` under `tardis_historical` and `kaiko_historical`. The pipeline
will skip Tier-A steps for nodes with missing paid archives and downgrade those nodes
to Tier-B automatically, logging a warning.

All paper claims are labelled by data tier, so Tier-A claims that require paid data are
clearly distinguished from Tier-B claims reproducible from free sources.
