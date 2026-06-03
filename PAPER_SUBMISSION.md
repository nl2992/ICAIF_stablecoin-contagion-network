# Paper Submission Snapshot

This file records the exact state of the repository at paper submission.
Fill in the fields below when tagging the submission commit.

---

## Submission metadata

| Field              | Value |
|--------------------|-------|
| Conference         | (fill in, e.g. ICAIF '26) |
| Submission ID      | (fill in from submission system) |
| Submission date    | (fill in, e.g. 2026-07-01) |
| Commit SHA         | (fill in: `git rev-parse HEAD`) |
| Branch at submit   | main |
| Tag                | (fill in: `git tag v0.1-icaif26-submit`) |

---

## Reproduction instructions

Starting from a clean checkout of the submission commit:

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Set API keys
cp .env.example .env
# Edit .env and set ETHERSCAN_API_KEY

# 3. Run the full empirical pipeline
make empirical_all

# 4. Build the paper package
make paper_gate

# 5. Validate
python scripts/14_validate_paper_package.py
```

The paper PDF is generated from `paper/main.tex`.
All claim-gated tables are in `results/paper/tables/`.
All figures are in `results/paper/figures/`.

---

## Optional dependencies (not required for primary results)

The Hawkes process analysis and the GNN are exploratory and not required
to reproduce the headline results.

```bash
# Hawkes (requires C++ compiler)
pip install -r requirements-optional.txt   # installs tick

# GNN (requires CUDA or CPU-only torch)
# See requirements-optional.txt for torch version
```

---

## What is and is not paper-claimable

The `table_claim_audit_summary.csv` in `results/paper/tables/` records every
edge tested, whether it passed the provenance gate, the statistical gate, and
the paper gate. Only rows with `paper_claim_allowed = True` appear in the
headline results.

The primary paper-claimable result is:
- Event: `usdt_curve_2023`
- Pair: `curve_3pool ↔ curve_crvusd_usdt`
- Feature: `usdc_net_sold_1h` at 3600s grid
- Result: Bonferroni-significant bidirectional lead-lag (p ≤ 0.014)
- Claim level: `A_A_dex_flow` / `claim_strength = robust`

All other events produce descriptive, suggestive, or contextual evidence only.
See `table_cross_event_comparison.csv` for the full breakdown.
