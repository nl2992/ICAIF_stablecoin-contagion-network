"""Forbes-Rigobon heteroskedasticity-corrected contagion test.

Tests whether stress-period correlation between two series is significantly
higher than tranquil-period correlation after correcting for the mechanical
inflation of correlations during high-volatility periods (Forbes & Rigobon 2002).

The null hypothesis is interdependence (shared volatility shift), not contagion.
Rejecting H0 supports a contagion claim.

Reference:
  Forbes, K. J., & Rigobon, R. (2002). No Contagion, Only Interdependence:
  Measuring Stock Market Comovements. Journal of Finance, 57(5), 2223–2261.

Tier note: outputs inherit the tier of the input feature (typically Tier A for
usdc_net_sold_1h). The contagion/interdependence classification is a paper claim
only when both input series are Tier A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FRResult:
    """Result for one directed pair under the Forbes-Rigobon test."""

    node_i: str
    node_j: str
    feature_col: str
    # Unconditional (tranquil + stress combined) correlation
    rho_unconditional: float
    # Raw stress-period correlation (before bias correction)
    rho_stress_raw: float
    # Forbes-Rigobon bias-corrected stress-period correlation
    rho_stress_corrected: float
    # Variance ratio: var_stress / var_tranquil for the conditioning series (node_i)
    delta: float
    # H0: rho_corrected == rho_tranquil; z-statistic and p-value (two-sided)
    z_stat: float
    p_value: float
    # n observations in each period
    n_tranquil: int
    n_stress: int
    # Classification
    contagion: bool       # True = significant difference after correction
    p_threshold: float    # threshold used


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = math.sqrt((x_c**2).sum() * (y_c**2).sum())
    if denom == 0:
        return float("nan")
    return float((x_c * y_c).sum() / denom)


def _fr_correct(rho_raw: float, delta: float) -> float:
    """Apply Forbes-Rigobon correction to a stress-period Pearson correlation.

    Corrected rho = rho_raw / sqrt(1 + delta*(1 - rho_raw**2))

    where delta = (var_stress - var_tranquil) / var_tranquil >= 0.
    """
    if not math.isfinite(rho_raw) or delta <= 0:
        return rho_raw
    denom = math.sqrt(1.0 + delta * (1.0 - rho_raw**2))
    if denom == 0:
        return rho_raw
    return rho_raw / denom


def _z_test(rho_a: float, rho_b: float, n_a: int, n_b: int) -> tuple[float, float]:
    """Fisher z-transform two-sample test for H0: rho_a == rho_b."""
    if not (math.isfinite(rho_a) and math.isfinite(rho_b)):
        return float("nan"), float("nan")
    # Clip to avoid log(0) at ±1
    rho_a = max(-0.9999, min(0.9999, rho_a))
    rho_b = max(-0.9999, min(0.9999, rho_b))
    z_a = 0.5 * math.log((1 + rho_a) / (1 - rho_a))
    z_b = 0.5 * math.log((1 + rho_b) / (1 - rho_b))
    se = math.sqrt(1.0 / max(n_a - 3, 1) + 1.0 / max(n_b - 3, 1))
    if se == 0:
        return float("nan"), float("nan")
    z_stat = (z_a - z_b) / se
    # Two-sided p-value via standard normal approximation
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z_stat)))
    return z_stat, p_value


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def run_forbes_rigobon(
    series_i: Sequence[float],
    series_j: Sequence[float],
    stress_mask: Sequence[bool],
    node_i: str = "node_i",
    node_j: str = "node_j",
    feature_col: str = "",
    p_threshold: float = 0.05,
) -> FRResult:
    """Run the Forbes-Rigobon test for one directed pair.

    Args:
        series_i:    Conditioning series (the potential contagion source).
        series_j:    Response series.
        stress_mask: Boolean mask; True = stress period.
        node_i:      Label for the source node.
        node_j:      Label for the response node.
        feature_col: Feature name (for traceability).
        p_threshold: Significance threshold for contagion classification.

    Returns:
        FRResult with corrected correlation and z-test.
    """
    xi = np.asarray(series_i, dtype=float)
    xj = np.asarray(series_j, dtype=float)
    mask = np.asarray(stress_mask, dtype=bool)

    tranquil_i = xi[~mask]
    tranquil_j = xj[~mask]
    stress_i   = xi[mask]
    stress_j   = xj[mask]

    n_tranquil = int((~mask).sum())
    n_stress   = int(mask.sum())

    rho_tranquil      = _pearson(tranquil_i, tranquil_j)
    rho_stress_raw    = _pearson(stress_i, stress_j)
    rho_unconditional = _pearson(xi, xj)

    var_tranquil = float(np.var(tranquil_i, ddof=1)) if n_tranquil > 1 else float("nan")
    var_stress   = float(np.var(stress_i,   ddof=1)) if n_stress > 1 else float("nan")

    if math.isfinite(var_tranquil) and var_tranquil > 0 and math.isfinite(var_stress):
        delta = max(0.0, (var_stress - var_tranquil) / var_tranquil)
    else:
        delta = 0.0

    rho_stress_corrected = _fr_correct(rho_stress_raw, delta)

    z_stat, p_value = _z_test(rho_stress_corrected, rho_tranquil, n_stress, n_tranquil)

    contagion = math.isfinite(p_value) and p_value < p_threshold

    return FRResult(
        node_i=node_i,
        node_j=node_j,
        feature_col=feature_col,
        rho_unconditional=rho_unconditional,
        rho_stress_raw=rho_stress_raw,
        rho_stress_corrected=rho_stress_corrected,
        delta=delta,
        z_stat=z_stat,
        p_value=p_value,
        n_tranquil=n_tranquil,
        n_stress=n_stress,
        contagion=contagion,
        p_threshold=p_threshold,
    )
