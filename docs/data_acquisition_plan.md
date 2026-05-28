# Data Acquisition Plan

This document specifies the ordered sequence of data fetch operations required to move
every P0-severity gap (see `docs/data_gaps.md`) from missing to Tier A.

---

## Priority order

### Step 1 — Tardis `USDCUSDT` Binance L2 (covers 3 events)

```bash
# Set TARDIS_API_KEY in .env first
python scripts/01b_ingest_vendor_l2.py \
    --exchange binance \
    --symbol USDCUSDT \
    --data-type incremental_book_L2 \
    --start 2023-03-08 --end 2023-03-14   # usdc_svb_2023
    --node-id usdc_binance_l2

python scripts/01b_ingest_vendor_l2.py \
    --exchange binance \
    --symbol USDCUSDT \
    --data-type incremental_book_L2 \
    --start 2023-06-08 --end 2023-06-18   # usdt_curve_2023
    --node-id usdt_binance_l2

python scripts/01b_ingest_vendor_l2.py \
    --exchange binance \
    --symbol USDCUSDT \
    --data-type incremental_book_L2 \
    --start 2022-11-06 --end 2022-11-16   # ftx_2022
    --node-id usdt_binance_l2
```

**Output:** `data/bronze/vendor_l2/binance_USDCUSDT_*.parquet`  
**Tier-A unlock:** `usdc_binance` and `usdt_binance` for three events.

---

### Step 2 — Tardis `BUSDUSDT` Binance L2 (covers busd_2023 + ftx_2022)

```bash
python scripts/01b_ingest_vendor_l2.py \
    --exchange binance \
    --symbol BUSDUSDT \
    --data-type incremental_book_L2 \
    --start 2023-02-10 --end 2023-02-18   # busd_2023

python scripts/01b_ingest_vendor_l2.py \
    --exchange binance \
    --symbol BUSDUSDT \
    --data-type incremental_book_L2 \
    --start 2022-11-06 --end 2022-11-16   # ftx_2022
```

---

### Step 3 — Tardis `USTUSDT` Binance (terra_luna_2022 — check availability first)

```bash
# Check if UST archive is available (Tardis retains delisted symbols)
python -c "
from stressnet.data.tardis_l2 import check_symbol_availability
print(check_symbol_availability('binance', 'USTUSDT', '2022-05-06', '2022-05-16'))
"

# If available:
python scripts/01b_ingest_vendor_l2.py \
    --exchange binance \
    --symbol USTUSDT \
    --data-type incremental_book_L2 \
    --start 2022-05-06 --end 2022-05-16   # terra_luna_2022
```

---

### Step 4 — L2 book reconstruction

Once bronze L2 files exist, run the reconstruction pipeline to produce
silver-level OHLCV + depth + bookwalk features:

```bash
python scripts/02b_reconstruct_l2_books.py --event usdc_svb_2023
python scripts/02b_reconstruct_l2_books.py --event usdt_curve_2023
python scripts/02b_reconstruct_l2_books.py --event ftx_2022
python scripts/02b_reconstruct_l2_books.py --event busd_2023
python scripts/02b_reconstruct_l2_books.py --event terra_luna_2022
```

Reconstructed features are merged into the feature panel by `scripts/03_build_feature_panel.py`
when it detects a `data/silver/l2_books/` directory for the given node.

---

### Step 5 — Dune DEX pool verification

Supplement The Graph subgraph data with verified on-chain Dune queries
for exact swap/liquidity event timestamps:

```bash
python scripts/01c_ingest_dune_queries.py \
    --query-id 3123456 \
    --event usdc_svb_2023 \
    --output curve_3pool_swaps
```

SQL templates: `sql/dune/curve_3pool_swaps.sql`, `sql/dune/uniswap_usdc_usdt_swaps.sql`

---

### Step 6 — Archive RPC for Uniswap exact block state (optional, if Dune insufficient)

```bash
# Set ETH_ARCHIVE_RPC_URL in .env
python scripts/01d_ingest_archive_pool_state.py \
    --pool uniswap_usdc_usdt_005 \
    --event usdc_svb_2023
```

---

## Vendor cost estimates (Tardis)

| Data type | Symbol | Window | Estimated compressed size |
|---|---|---|---|
| `incremental_book_L2` | `USDCUSDT` | 7 days | ~8 GB |
| `incremental_book_L2` | `BUSDUSDT` | 7 days | ~3 GB |
| `incremental_book_L2` | `USTUSDT` | 10 days | ~5 GB (if available) |
| `book_snapshot_25` | `USDC-USD` (Coinbase) | 7 days | ~2 GB |

Tardis free tier does **not** include historical data.  A paid subscription or
`TARDIS_API_KEY` from the academic programme is required.  See:
https://docs.tardis.dev/api/tardis-machine

---

## Kaiko alternative (if Tardis unavailable)

Kaiko provides equivalent `full_order_book` data.  Set `DATA_VENDOR=kaiko` in `.env`
and the ingestion scripts will route to `src/stressnet/data/kaiko_l2.py` instead.

---

## Checkpoint: when to re-run the claim gate

After each step above, re-run:

```bash
python scripts/00d_check_empirical_coverage.py --event <event> --require-tier-a-nodes 2
python scripts/00c_claim_gate.py --event <event> --strict
```

A passing strict-mode gate confirms that at least one A/A edge exists for the event.
