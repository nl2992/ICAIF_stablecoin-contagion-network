.PHONY: setup test windows coverage audit claimgate ingest reconstruct panel eventmaps maps leadlag var tvpvar hawkes te network predict predset gnn eventstudy robustness summary paper empirical empirical_all paper_gate mvp usdc all

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests -q

# --- pipeline steps (default: USDC/SVB) ---

EVENT ?= usdc_svb_2023
GRID ?= 60
INGEST_FLAGS ?=

windows:
	python scripts/00_make_event_windows.py

coverage:
	python scripts/00_make_event_windows.py --coverage-audit

audit:
	python scripts/00b_audit_provenance.py --event $(EVENT)

claimgate:
	python scripts/00c_claim_gate.py --event $(EVENT)

ingest:
	python scripts/01_ingest_raw_data.py --event $(EVENT) $(INGEST_FLAGS)

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
	python scripts/05b_run_tvp_var.py --event $(EVENT) --window-size 168 --step-size 24

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

eventstudy:
	python scripts/12_run_event_study.py --event $(EVENT)

robustness:
	python scripts/10_run_robustness.py --event $(EVENT)

summary:
	python scripts/11b_summarise_real_only_results.py
	python scripts/11c_summarise_robustness.py
	python scripts/11_summarise_results.py

paper:
	python scripts/99_make_paper_outputs.py

# paper_gate: annotate ALL events, write claim-gated tables to results/paper/tables/
# Fails (exit 1) if any paper-path table contains fixture-derived edges.
paper_gate:
	python scripts/00c_claim_gate.py --all-events --strict
	python scripts/99_make_paper_outputs.py

# run an empirical paper-claim pipeline for one event. This target disables
# fixture fallback and gates result edges by endpoint provenance before paper use.
empirical:
	python scripts/00_make_event_windows.py
	python scripts/01_ingest_raw_data.py --event $(EVENT) --no-fixture
	python scripts/02_reconstruct_silver.py --event $(EVENT)
	python scripts/03_build_feature_panel.py --event $(EVENT) --grid $(GRID)
	python scripts/03b_make_event_maps.py --event $(EVENT)
	python scripts/04_run_leadlag.py --event $(EVENT) --paper-mode
	python scripts/05_run_var_granger.py --event $(EVENT)
	python scripts/05b_run_tvp_var.py --event $(EVENT) --paper-mode --window-size 168 --step-size 24
	python scripts/06_run_hawkes.py --event $(EVENT) || true
	python scripts/07_run_transfer_entropy.py --event $(EVENT) --paper-mode
	python scripts/08_build_networks.py --event $(EVENT) --edge-source te
	python scripts/09_run_prediction.py --event $(EVENT)
	python scripts/10_run_robustness.py --event $(EVENT)
	python scripts/12_run_event_study.py --event $(EVENT) --paper-mode
	python scripts/00c_claim_gate.py --event $(EVENT)

empirical_all:
	for event in usdc_svb_2023 terra_luna_2022 usdt_curve_2023 ftx_2022 busd_2023; do \
		$(MAKE) empirical EVENT=$$event GRID=$(GRID); \
	done
	$(MAKE) paper_gate

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
