.PHONY: setup test windows coverage panel leadlag var hawkes te network predict robustness paper all

setup:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests -q

# --- pipeline steps (default: USDC/SVB) ---

EVENT ?= usdc_svb_2023

windows:
	python scripts/00_make_event_windows.py

coverage:
	python scripts/00_make_event_windows.py --coverage-audit

panel:
	python scripts/03_build_feature_panel.py --event $(EVENT)

leadlag:
	python scripts/04_run_leadlag.py --event $(EVENT)

var:
	python scripts/05_run_var_granger.py --event $(EVENT)

hawkes:
	python scripts/06_run_hawkes.py --event $(EVENT)

te:
	python scripts/07_run_transfer_entropy.py --event $(EVENT)

network:
	python scripts/08_build_networks.py --event $(EVENT)

predict:
	python scripts/09_run_prediction.py --event $(EVENT)

robustness:
	python scripts/10_run_robustness.py --event $(EVENT)

paper:
	python scripts/99_make_paper_outputs.py

# run full analysis pipeline for one event
mvp: panel leadlag var hawkes te network

# run all events sequentially
all:
	for event in usdc_svb_2023 terra_luna_2022 usdt_curve_2023 ftx_2022 busd_2023; do \
		$(MAKE) mvp EVENT=$$event; \
	done
	$(MAKE) paper
