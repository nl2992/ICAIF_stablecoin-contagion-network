"""Tests for Uniswap v3 Etherscan Swap-log ingestion."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from stressnet.data.uniswap_etherscan import (
    TOPIC_UNISWAP_V3_SWAP,
    decode_uniswap_swap,
    ingest_uniswap_pool_events,
)

POOL_USDC_USDT = "0x3416cf6c708da44db2624d63ea0aaef7113527c6"


def _abi_int(value: int) -> str:
    return value.to_bytes(32, "big", signed=value < 0).hex()


def _swap_data(amount0: int, amount1: int, sqrt_price: int = 1, liquidity: int = 1, tick: int = 0) -> str:
    return "0x" + "".join(
        [_abi_int(amount0), _abi_int(amount1), _abi_int(sqrt_price), _abi_int(liquidity), _abi_int(tick)]
    )


def test_decode_uniswap_swap_uses_token_decimals_and_sign_convention():
    data = _swap_data(-1_500_000, 1_500_000)

    decoded = decode_uniswap_swap(data, POOL_USDC_USDT)

    assert decoded is not None
    assert decoded["token0"] == "USDC"
    assert decoded["token1"] == "USDT"
    assert decoded["amount0"] == -1.5
    assert decoded["amount1"] == 1.5
    assert decoded["usdc_net_sold"] == 1.5


def test_ingest_uniswap_pool_events_aggregates_swap_logs(monkeypatch, tmp_path):
    logs = [
        {
            "blockNumber": "0x10",
            "logIndex": "0x0",
            "timeStamp": hex(int(datetime(2023, 6, 15, 0, 1, tzinfo=timezone.utc).timestamp())),
            "data": _swap_data(-1_000_000, 1_000_000),
        },
        {
            "blockNumber": "0x11",
            "logIndex": "0x0",
            "timeStamp": hex(int(datetime(2023, 6, 15, 0, 5, tzinfo=timezone.utc).timestamp())),
            "data": _swap_data(-2_000_000, 2_000_000),
        },
    ]

    import stressnet.data.uniswap_etherscan as uni

    monkeypatch.setenv("ETHERSCAN_API_KEY", "test-key")
    monkeypatch.setattr(uni, "get_all_logs", lambda *args, **kwargs: logs)

    out_path, tier = ingest_uniswap_pool_events(
        POOL_USDC_USDT,
        1,
        2,
        tmp_path,
        "usdt_curve_2023",
        "uniswap_usdc_usdt_005",
        grid_seconds=3600,
    )

    assert tier == "A"
    assert out_path is not None
    df = pl.read_parquet(out_path)
    assert df.height == 1
    assert df["usdc_net_sold_1h"][0] == 3.0
    assert df["n_events"][0] == 2
    assert "reserve_imbalance" in df.columns
