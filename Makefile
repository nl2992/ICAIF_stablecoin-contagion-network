.PHONY: setup test windows coverage audit ingest reconstruct panel eventmaps maps leadlag var tvpvar hawkes te network predict predset gnn robustness paper mvp usdc all

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests -q

# --- pipeline steps (default: USDC/SVB) ---

EVENT ?= usdc_svb_2023
GRID ?= 60

windows:
	python scripts/00_make_event_windows.py

coverage:
	python scripts/00_make_event_windows.py --coverage-audit

audit:
	python scripts/00b_audit_provenance.py --event $(EVENT)

ingest:
	python scripts/01_ingest_raw_data.py --event $(EVENT)

reconstruct:
	python scripts/02_reconstruct_silver.py --event $(EVENT)

panel:
	python scripts/03_build_feature_panel.py --event $(EVENT) --grid $(GRID)

eventmaps:
	python scripts/03b_make_event_maps.py --event $(EVENT)

maps: eventmaps

leadlag:
	python scripts/04_run_leadlag.py --event $(EVENT)

var:
	python scripts/05_run_var_granger.py --event $(EVENT)

tvpvar:
	python scripts/05b_run_tvp_var.py --event $(EVENT)

hawkes:
	python scripts/06_run_hawkes.py --event $(EVENT)

te:
	python scripts/07_run_transfer_entropy.py --event $(EVENT)

network:
	python scripts/08_build_networks.py --event $(EVENT) --edge-source te

predict:
	python scripts/09_run_prediction.py --event $(EVENT)

predset:
	python scripts/09b_make_prediction_dataset.py --event $(EVENT)

gnn:
	python scripts/09d_train_temporal_gnn.py --event $(EVENT)

robustness:
	python scripts/10_run_robustness.py --event $(EVENT)

paper:
	python scripts/99_make_paper_outputs.py

# run empirical-control pipeline for one event. Hawkes is optional until its
# dependency is installed and configured.
mvp: windows ingest reconstruct panel eventmaps leadlag var te network

usdc:
	$(MAKE) mvp EVENT=usdc_svb_2023

# run all events sequentially
all:
	for event in usdc_svb_2023 terra_luna_2022 usdt_curve_2023 ftx_2022 busd_2023; do \
		$(MAKE) mvp EVENT=$$event; \
	done
	$(MAKE) paper
