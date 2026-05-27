"""Tests for DEX pool state reconstruction."""

import pytest
from stressnet.reconstruct.dex_pool import CurvePoolState, UniswapV3PoolState


def test_curve_pool_balanced():
    state = CurvePoolState(reserves=[1e6, 1e6, 1e6])
    imb = state.reserve_imbalance(token_idx=0)
    assert imb == pytest.approx(0.0, abs=1e-9)


def test_curve_pool_imbalanced():
    state = CurvePoolState(reserves=[2e6, 1e6, 1e6])
    imb = state.reserve_imbalance(token_idx=0)
    assert imb > 0  # token_0 is over-represented


def test_curve_implied_price():
    state = CurvePoolState(reserves=[1e6, 1e6])
    price = state.implied_price(i=0, j=1)
    assert price == pytest.approx(1.0)


def test_curve_slippage_positive():
    state = CurvePoolState(reserves=[1e8, 1e8, 1e8])
    slippage = state.slippage_bps(10_000)
    assert slippage is not None
    assert slippage > 0


def test_uniswap_v3_implied_price():
    # sqrtPriceX96 for price ≈ 1.0: sqrt(1) * 2**96
    sqrt_p = int(1.0 * (2**96))
    state = UniswapV3PoolState(sqrt_price_x96=sqrt_p)
    price = state.implied_price()
    assert price == pytest.approx(1.0, rel=1e-4)


def test_uniswap_tick_price():
    state = UniswapV3PoolState(tick=0)
    assert state.tick_to_price(0) == pytest.approx(1.0)
    assert state.tick_to_price(10000) > 1.0


def test_curve_from_event():
    event = {
        "reserves": [1e6, 1e6, 1e6],
        "A": 100,
        "virtual_price": 1.001,
        "lp_supply": 3e6,
        "block_number": 17_000_000,
        "block_ts": 1678000000,
    }
    state = CurvePoolState.from_event(event)
    assert state.virtual_price == pytest.approx(1.001)
    assert state.block_number == 17_000_000
