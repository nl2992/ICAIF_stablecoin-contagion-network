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

# 2. ingest raw/bronze data for one event
python scripts/01_ingest_raw_data.py --event usdc_svb_2023
# → data/raw/ (not committed; hashes in data/manifests/)
# → data/bronze/

# 3. reconstruct silver books, pools, and flows
python scripts/02_reconstruct_silver.py --event usdc_svb_2023
# → data/silver/

# 4. build feature panel
python scripts/03_build_feature_panel.py --event usdc_svb_2023
# → data/gold/dataset_contagion_features_usdc_svb_2023.parquet

# 5–8. analysis (pipeline demo / MVP)
make mvp EVENT=usdc_svb_2023

# 9. provenance-filtered summaries
make claimgate EVENT=usdc_svb_2023
make summary

# 10. paper outputs
make paper
```

The default `make mvp` target is allowed to use deterministic fixtures when real
public data is unavailable. These fixture rows are tagged as
`fixture_non_empirical` and are only for validating orchestration, schemas,
manifests, plots, and table plumbing.

Paper-claim runs should use the empirical target:

```bash
make empirical EVENT=usdc_svb_2023 GRID=60
```

This target passes `--no-fixture` to ingestion and runs the claim gate after
estimation. `make paper` also runs `scripts/00c_claim_gate.py --paper
--require-real`, so it refuses to build paper outputs from edge tables that use
fixture or missing endpoint tiers.

Once every configured event has sufficient real data coverage, run the full
empirical benchmark with:

```bash
make empirical_all GRID=60
```

`make empirical_all` runs the empirical target for all five configured events
and then invokes `make paper`.

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
clearly distinguished from Tier-B claims reproducible from free sources. Edge-level
claims are capped by the weaker endpoint tier and blocked entirely when either endpoint
is `fixture_non_empirical` or missing from provenance.
