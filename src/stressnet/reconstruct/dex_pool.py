"""DEX pool state reconstruction from on-chain event logs.

Supports Curve (StableSwap invariant) and Uniswap v3 (concentrated liquidity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurvePoolState:
    """Reconstructed state of a Curve StableSwap pool.

    Attributes:
        reserves: Token reserves, e.g. [dai_reserve, usdc_reserve, usdt_reserve].
        amplification: Current A parameter.
        virtual_price: Pool virtual_price() value; > 1.0 indicates LP fee accrual.
        lp_supply: Total LP token supply.
        fee_bps: Pool swap fee in basis points.
        block_number: Block at which this state was snapshotted.
        block_ts: Block timestamp (Unix seconds).
    """

    reserves: list[float] = field(default_factory=list)
    amplification: float = 100.0
    virtual_price: float = 1.0
    lp_supply: float = 0.0
    fee_bps: float = 4.0
    block_number: int = 0
    block_ts: int = 0

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "CurvePoolState":
        """Construct pool state from a decoded on-chain event dict."""
        return cls(
            reserves=event.get("reserves", []),
            amplification=float(event.get("A", 100)),
            virtual_price=float(event.get("virtual_price", 1)),
            lp_supply=float(event.get("lp_supply", 0)),
            fee_bps=float(event.get("fee_bps", 4)),
            block_number=int(event.get("block_number", 0)),
            block_ts=int(event.get("block_ts", 0)),
        )

    def reserve_imbalance(self, token_idx: int = 0) -> float | None:
        """Fraction of pool in token ``token_idx`` minus equal-weight fraction.

        Positive = over-represented; negative = under-represented.
        """
        if not self.reserves or sum(self.reserves) == 0:
            return None
        total = sum(self.reserves)
        n = len(self.reserves)
        return self.reserves[token_idx] / total - 1.0 / n

    def implied_price(self, i: int = 0, j: int = 1) -> float | None:
        """Approximate marginal price of token i in terms of token j.

        Uses the first-order StableSwap approximation around the current reserve ratio.
        For precise on-chain prices, prefer sqrtPriceX96 from Uniswap v3 events.
        """
        if not self.reserves or len(self.reserves) <= max(i, j):
            return None
        if self.reserves[i] <= 0:
            return None
        return self.reserves[j] / self.reserves[i]

    def slippage_bps(self, notional_usd: float, token_in_idx: int = 0) -> float | None:
        """Estimate price impact in bps for a swap of ``notional_usd`` USD.

        Uses a simplified linear approximation; sufficient for feature computation.
        """
        if not self.reserves:
            return None
        reserve_in = self.reserves[token_in_idx]
        if reserve_in <= 0:
            return None
        impact = notional_usd / (reserve_in * 2)
        return impact * 10_000


@dataclass
class UniswapV3PoolState:
    """Reconstructed state of a Uniswap v3 concentrated-liquidity pool.

    Attributes:
        sqrt_price_x96: Current sqrtPriceX96 from the pool slot0.
        tick: Current active tick.
        liquidity: Active liquidity at current tick.
        fee_bps: Pool fee in basis points (5 = 0.05%).
        block_number: Block at which this state was snapshotted.
        block_ts: Block timestamp (Unix seconds).
    """

    sqrt_price_x96: int = 0
    tick: int = 0
    liquidity: int = 0
    fee_bps: float = 5.0
    block_number: int = 0
    block_ts: int = 0

    @classmethod
    def from_swap_event(cls, event: dict[str, Any]) -> "UniswapV3PoolState":
        """Construct pool state from a decoded Swap event."""
        return cls(
            sqrt_price_x96=int(event.get("sqrtPriceX96", 0)),
            tick=int(event.get("tick", 0)),
            liquidity=int(event.get("liquidity", 0)),
            fee_bps=float(event.get("fee_bps", 5)),
            block_number=int(event.get("block_number", 0)),
            block_ts=int(event.get("block_ts", 0)),
        )

    def implied_price(self) -> float | None:
        """Compute the implied price from sqrtPriceX96.

        price = (sqrtPriceX96 / 2**96) ** 2
        For USDC/USDT pools both tokens have 6 decimals, so no decimal adjustment needed.
        """
        if self.sqrt_price_x96 <= 0:
            return None
        return (self.sqrt_price_x96 / (2**96)) ** 2

    def tick_to_price(self, tick: int | None = None) -> float:
        """Convert a tick to price: price = 1.0001^tick."""
        t = tick if tick is not None else self.tick
        return 1.0001 ** t
