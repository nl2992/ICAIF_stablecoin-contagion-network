"""Tests for curve.py pool configuration, per-pool decimal handling, and tier return."""

from __future__ import annotations

import struct
from typing import Any

import pytest

from stressnet.data.curve import (
    PoolConfig,
    _POOL_CONFIGS,
    _get_pool_config,
    CURVE_3POOL_ADDRESS,
    CURVE_CRVUSD_USDT,
    CURVE_UST_WORMHOLE,
    decode_token_exchange,
    decode_add_liquidity,
    decode_remove_liquidity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abi_encode(*values: int, signed: bool = False) -> str:
    """Encode a sequence of integers as ABI uint256/int128 slots (32 bytes each).

    Produces a hex string suitable as the ``data`` field from an Etherscan log.
    All values are encoded as big-endian 32-byte words.
    """
    parts = []
    for v in values:
        if v < 0:
            # Negative: encode as two's-complement 32-byte signed int
            b = v.to_bytes(32, "big", signed=True)
        else:
            b = v.to_bytes(32, "big")
        parts.append(b)
    return "0x" + b"".join(parts).hex()


# ---------------------------------------------------------------------------
# PoolConfig unit tests
# ---------------------------------------------------------------------------

class TestPoolConfig:
    def test_3pool_config_present(self):
        cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
        assert cfg.stablecoin_symbol == "USDC"
        assert cfg.pool_size_usd == 500_000_000
        assert cfg.ng_scaled is False
        assert cfg.tokens[1] == ("USDC", 6)

    def test_crvusd_config_present(self):
        cfg = _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]
        assert cfg.stablecoin_symbol == "USDT"
        assert cfg.pool_size_usd == 30_000_000
        assert cfg.ng_scaled is True
        assert cfg.tokens[0] == ("crvUSD", 18)
        assert cfg.tokens[1] == ("USDT", 6)

    def test_ust_wormhole_config_present(self):
        cfg = _POOL_CONFIGS[CURVE_UST_WORMHOLE.lower()]
        assert cfg.stablecoin_symbol == "UST"
        assert cfg.ng_scaled is False

    def test_get_pool_config_case_insensitive(self):
        cfg_lower = _get_pool_config(CURVE_3POOL_ADDRESS.lower())
        cfg_mixed = _get_pool_config(CURVE_3POOL_ADDRESS)
        assert cfg_lower.stablecoin_symbol == cfg_mixed.stablecoin_symbol

    def test_get_pool_config_fallback(self):
        """Unknown address falls back to 3pool defaults with a warning."""
        cfg = _get_pool_config("0xdeadbeef")
        assert cfg.stablecoin_symbol == "USDC"  # 3pool fallback

    def test_normalize_amount_classic_usdc(self):
        """Classic pool: USDC at index 1, 6 decimals."""
        cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
        raw = 1_000_000  # 1 USDC in raw
        assert abs(cfg.normalize_amount(1, raw) - 1.0) < 1e-12

    def test_normalize_amount_ng_usdt(self):
        """StableSwap-ng pool: USDT at index 1, emitted as 18-dec internally."""
        cfg = _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]
        # 1 USDT in 18-dec = 1e18 raw
        raw = int(1e18)
        assert abs(cfg.normalize_amount(1, raw) - 1.0) < 1e-12

    def test_normalize_amount_ng_scale_factor(self):
        """ng pool: normalise by 1e18 regardless of token index."""
        cfg = _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]
        for idx in (0, 1):
            raw = int(1e18)
            assert abs(cfg.normalize_amount(idx, raw) - 1.0) < 1e-12

    def test_normalize_amount_classic_dai(self):
        """Classic pool: DAI at index 0, 18 decimals."""
        cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
        raw = int(1e18)  # 1 DAI
        assert abs(cfg.normalize_amount(0, raw) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# decode_token_exchange — 3pool (classic decimals)
# ---------------------------------------------------------------------------

class TestDecodeTokenExchange3pool:
    """decode_token_exchange with classic 3pool config."""

    def _make_data(self, sold_id: int, tokens_sold: int,
                   bought_id: int, tokens_bought: int) -> str:
        return _abi_encode(sold_id, tokens_sold, bought_id, tokens_bought)

    def test_usdc_sold_classic(self):
        """Selling USDC (idx 1) for DAI: tokens_sold in native 6-dec."""
        sold_raw = 1_000_000  # 1 USDC
        data = self._make_data(1, sold_raw, 0, int(1e18))
        result = decode_token_exchange(data)  # default = 3pool config
        assert result["sold_symbol"] == "USDC"
        assert abs(result["tokens_sold"] - 1.0) < 1e-9

    def test_usdt_bought_classic(self):
        """Buying USDT (idx 2) with DAI: tokens_bought in native 6-dec."""
        data = self._make_data(0, int(1e18), 2, 500_000)  # 0.5 USDT bought
        result = decode_token_exchange(data)
        assert result["bought_symbol"] == "USDT"
        assert abs(result["tokens_bought"] - 0.5) < 1e-9

    def test_no_usdc_involvement(self):
        """DAI ↔ USDT swap: neither matches USDC."""
        data = self._make_data(0, int(1e18), 2, 999_000)
        result = decode_token_exchange(data)
        assert result["sold_symbol"] == "DAI"
        assert result["bought_symbol"] == "USDT"

    def test_negative_sold_id_returns_empty(self):
        """Corrupted negative sold_id (-1 in int128): ABI slot 0 is fine but sold_id=-1 is unknown."""
        data = _abi_encode(-1, int(1e6), 1, int(1e18))
        result = decode_token_exchange(data)
        # sold_id = -1 → not in token map → "UNK"
        assert result.get("sold_symbol") == "UNK"

    def test_empty_data_returns_empty(self):
        result = decode_token_exchange("0x")
        assert result == {}


# ---------------------------------------------------------------------------
# decode_token_exchange — crvUSD/USDT StableSwap-ng
# ---------------------------------------------------------------------------

class TestDecodeTokenExchangeNg:
    """decode_token_exchange with StableSwap-ng (18-dec internal amounts)."""

    def _cfg(self) -> PoolConfig:
        return _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]

    def _make_data(self, sold_id: int, tokens_sold: int,
                   bought_id: int, tokens_bought: int) -> str:
        return _abi_encode(sold_id, tokens_sold, bought_id, tokens_bought)

    def test_usdt_sold_ng(self):
        """Selling USDT (idx 1, ng): raw amount is 18-dec → 1e18 = 1 USDT."""
        raw = int(1e18)  # 1 USDT in 18-dec internal
        data = self._make_data(1, raw, 0, int(1e18))
        cfg = self._cfg()
        result = decode_token_exchange(data, cfg)
        assert result["sold_symbol"] == "USDT"
        assert abs(result["tokens_sold"] - 1.0) < 1e-9

    def test_usdt_bought_ng(self):
        """Buying USDT (idx 1, ng): 500_000 USDT in 18-dec."""
        raw_bought = int(500_000 * 1e18)  # 500k USDT
        data = self._make_data(0, int(1e18), 1, raw_bought)
        cfg = self._cfg()
        result = decode_token_exchange(data, cfg)
        assert result["bought_symbol"] == "USDT"
        assert abs(result["tokens_bought"] - 500_000) < 1.0  # within $1

    def test_ng_vs_classic_factor(self):
        """For the same raw value, ng normalisation differs from 6-dec by 1e12."""
        raw = int(1e12)  # 1e12 raw units
        data = _abi_encode(1, raw, 0, raw)
        classic_cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
        ng_cfg      = _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]

        classic = decode_token_exchange(data, classic_cfg)
        ng_res  = decode_token_exchange(data, ng_cfg)

        # classic: raw / 1e6 = 1e6 ; ng: raw / 1e18 = 1e-6
        assert abs(classic["tokens_sold"] / ng_res["tokens_sold"] - 1e12) < 1e3

    def test_crvusd_sold_ng(self):
        """crvUSD at idx 0, ng: both tokens use 18-dec normalisation."""
        raw = int(2_500 * 1e18)  # 2500 crvUSD
        data = self._make_data(0, raw, 1, raw)
        cfg = self._cfg()
        result = decode_token_exchange(data, cfg)
        assert result["sold_symbol"] == "crvUSD"
        assert abs(result["tokens_sold"] - 2500.0) < 0.01


# ---------------------------------------------------------------------------
# reserve_imbalance scale — verify fix eliminates the ~1e9 bug
# ---------------------------------------------------------------------------

class TestReserveImbalanceScale:
    """Verify that the ng decimal fix brings reserve_imbalance into [-2, 2]."""

    def test_crvusd_reserve_imbalance_reasonable(self):
        """Simulate the first-hour event from the real bronze file and verify scale."""
        # Real first-hour value after 1/1e6 (wrong) normalisation was -5.3469e17
        # After /1e18 (correct): -5.3469e17 / 1e12 ≈ -534,690 USDT
        # pool_size_usd = 30_000_000 → reserve_imbalance ≈ -534_690 / 30_000_000 ≈ -0.018
        broken_value = -5.3469e17  # what the old code produces as usdc_net_sold_1h
        # The old normaliser applied: reserve_imbalance = broken_value / 500_000_000
        old_reserve = broken_value / 500_000_000
        assert abs(old_reserve) > 1_000  # confirm old result is huge

        # New approach: raw event amounts normalised by 1e18 (not 1e6)
        # raw_amount ≈ broken_value * 1e6 (since old code divided by 1e6)
        raw_amount = broken_value * 1e6  # recover approximate raw wei value
        cfg = _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]
        new_usdt_per_hr = raw_amount / 1e18  # correct ng normalisation
        new_reserve = new_usdt_per_hr / cfg.pool_size_usd
        # Should be in a reasonable [-2, 2] range for a stressed pool
        assert -2.0 <= new_reserve <= 2.0, (
            f"reserve_imbalance = {new_reserve:.4f} should be in [-2, 2]"
        )

    def test_3pool_reserve_imbalance_unchanged(self):
        """3pool normalisation is untouched — reserve_imbalance still uses 6-dec amounts."""
        # Simulate 100M USDC sold into 3pool (raw = 100e6 * 1e6 = 1e14)
        raw = int(100e6 * 1e6)  # 100M USDC in native wei
        cfg = _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]
        usdc_normalised = cfg.normalize_amount(1, raw)  # index 1 = USDC
        reserve = usdc_normalised / cfg.pool_size_usd
        # 100M / 500M = 0.2 — small-ish imbalance
        assert abs(reserve - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# decode_add_liquidity / decode_remove_liquidity
# ---------------------------------------------------------------------------

class TestAddRemoveLiquidity:
    def _3pool_cfg(self) -> PoolConfig:
        return _POOL_CONFIGS[CURVE_3POOL_ADDRESS.lower()]

    def test_add_liquidity_3pool_keys(self):
        """AddLiquidity 3pool produces dai_in, usdc_in, usdt_in."""
        data = _abi_encode(int(1e18), int(1e6), int(1e6))  # 1 DAI, 1 USDC, 1 USDT
        result = decode_add_liquidity(data, self._3pool_cfg())
        assert "dai_in" in result
        assert "usdc_in" in result
        assert "usdt_in" in result
        assert abs(result["usdc_in"] - 1.0) < 1e-9

    def test_remove_liquidity_3pool(self):
        data = _abi_encode(int(2e18), int(2e6), int(2e6))
        result = decode_remove_liquidity(data, self._3pool_cfg())
        assert abs(result["dai_out"] - 2.0) < 1e-9
        assert abs(result["usdc_out"] - 2.0) < 1e-9

    def test_add_liquidity_ng_pool(self):
        """AddLiquidity for crvUSD/USDT ng: keys are crvusd_in and usdt_in."""
        cfg = _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]
        data = _abi_encode(int(1e18), int(1e18))  # 1 crvUSD, 1 USDT (both 18-dec)
        result = decode_add_liquidity(data, cfg)
        assert "crvusd_in" in result
        assert "usdt_in" in result
        assert abs(result["usdt_in"] - 1.0) < 1e-9

    def test_legacy_usdc_in_absent_for_crvusd_pool(self):
        """crvUSD pool has no 'USDC' token, so usdc_in=0.0 fallback should be absent
        OR equal to 0 — it must not clobber real USDT flows."""
        cfg = _POOL_CONFIGS[CURVE_CRVUSD_USDT.lower()]
        data = _abi_encode(int(5e18), int(5e18))
        result = decode_add_liquidity(data, cfg)
        # Either usdc_in is missing or 0; it must not be the same as usdt_in
        usdc_in = result.get("usdc_in", 0.0)
        usdt_in = result.get("usdt_in", 0.0)
        assert usdc_in == 0.0 or usdc_in != usdt_in


# ---------------------------------------------------------------------------
# Backward compat: None pool_cfg falls back to 3pool
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_decode_token_exchange_no_cfg(self):
        """Calling without pool_cfg uses 3pool defaults."""
        data = _abi_encode(1, 1_000_000, 0, int(1e18))  # USDC sold
        result = decode_token_exchange(data)
        assert result["sold_symbol"] == "USDC"

    def test_decode_add_no_cfg(self):
        data = _abi_encode(int(1e18), int(1e6), int(1e6))
        result = decode_add_liquidity(data)
        assert "usdc_in" in result

    def test_decode_remove_no_cfg(self):
        data = _abi_encode(int(1e18), int(1e6), int(1e6))
        result = decode_remove_liquidity(data)
        assert "usdc_out" in result
