.PHONY: setup test windows coverage coveragegate audit claimgate ingest reconstruct panel eventmaps combine maps leadlag hy var tvpvar hawkes te network predict predset gnn eventstudy robustness summary paper paper_gate empirical empirical_all mvp usdc demo_all all

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

coveragegate:
	python scripts/00d_check_empirical_coverage.py --event $(EVENT) --require-layers CEX DEX

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

combine:
	python scripts/03c_combine_panels.py

maps: eventmaps

leadlag:
	python scripts/04_run_leadlag.py --event $(EVENT)

hy:
	python scripts/04b_run_hayashi_yoshida.py --event $(EVENT)

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

# Diagnostic paper build (reads results/tables, no claim enforcement)
paper:
	python scripts/99_make_paper_outputs.py

# Claim-gated paper build: annotate all events, strict-exit on fixture,
# then assemble final paper outputs exclusively from results/paper/tables/.
paper_gate:
	python scripts/00c_claim_gate.py --all-events --strict
	python scripts/99_make_paper_outputs.py --strict

# run an empirical paper-claim pipeline for one event.
# Disables fixture fallback; gates result edges by provenance; uses --paper-mode
# for all analysis scripts so only real nodes enter the model.
empirical:
	python scripts/00_make_event_windows.py
	python scripts/01_ingest_raw_data.py --event $(EVENT) --no-fixture
	python scripts/02_reconstruct_silver.py --event $(EVENT)
	python scripts/03_build_feature_panel.py --event $(EVENT) --grid $(GRID)
	python scripts/00d_check_empirical_coverage.py --event $(EVENT) --require-layers CEX DEX
	python scripts/03b_make_event_maps.py --event $(EVENT)
	python scripts/04_run_leadlag.py --event $(EVENT) --paper-mode
	python scripts/04b_run_hayashi_yoshida.py --event $(EVENT) --paper-mode
	python scripts/05_run_var_granger.py --event $(EVENT)
	python scripts/05b_run_tvp_var.py --event $(EVENT) --paper-mode --window-size 168 --step-size 24
	python scripts/06_run_hawkes.py --event $(EVENT) || true
	python scripts/07_run_transfer_entropy.py --event $(EVENT) --paper-mode
	python scripts/08_build_networks.py --event $(EVENT) --edge-source te
	python scripts/09_run_prediction.py --event $(EVENT)
	python scripts/10_run_robustness.py --event $(EVENT)
	python scripts/12_run_event_study.py --event $(EVENT) --paper-mode
	python scripts/00c_claim_gate.py --event $(EVENT)

# Run all 5 events empirically then assemble claim-gated paper outputs.
empirical_all:
	for event in usdc_svb_2023 terra_luna_2022 usdt_curve_2023 ftx_2022 busd_2023; do \
		$(MAKE) empirical EVENT=$$event GRID=$(GRID); \
	done
	for event in usdc_svb_2023 terra_luna_2022 usdt_curve_2023 ftx_2022 busd_2023; do \
		python scripts/09_run_prediction.py --event $$event --loeo --ablation; \
	done
	$(MAKE) paper_gate

# MVP demo: fixture-allowed, single event, no claim gate (for orchestration testing only)
mvp: windows ingest reconstruct panel eventmaps leadlag var te network

usdc:
	$(MAKE) mvp EVENT=usdc_svb_2023

# demo_all: run mvp (fixture-allowed) for all events + diagnostic paper build.
# WARNING: outputs may contain fixture-derived edges; do NOT use for paper claims.
demo_all:
	for event in usdc_svb_2023 terra_luna_2022 usdt_curve_2023 ftx_2022 busd_2023; do \
		$(MAKE) mvp EVENT=$$event; \
	done
	$(MAKE) paper

# all: canonical paper-safe target. Runs empirical_all (real data only).
all: empirical_all
