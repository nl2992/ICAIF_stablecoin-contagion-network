"""Quantile VAR (QVAR) — rolling pairwise quantile regression impulse responses.

Estimates a VAR at three quantiles (τ=0.05, 0.50, 0.95) using statsmodels
QuantReg to capture tail-spillover asymmetry.  The key finding this supports is
that tail spillovers (τ=0.05, τ=0.95) are materially larger than median
spillovers (τ=0.50), consistent with non-linear contagion dynamics.

Method:
  For each ordered pair (i, j) at each quantile τ:
    Q_τ(x_j,t | x_j,t-1, x_i,t-1) = α + β_own * x_j,t-1 + β_cross * x_i,t-1
  The cross-coefficient β_cross is the quantile impulse-response estimate.

Tier note: outputs inherit the feature tier of the input series.  For
usdc_net_sold_1h (Tier A), QVAR results are paper-claimable at A/A level when
both nodes are Tier A.

Reference:
  Koenker, R. & Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1).
  White, H., Kim, T., & Manganelli, S. (2015). VAR for VaR: Measuring tail
  dependence using multivariate regression quantiles. Journal of Econometrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import statsmodels.formula.api as smf
    import pandas as pd
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


_DEFAULT_QUANTILES = (0.05, 0.50, 0.95)
_DEFAULT_LAGS = 1


@dataclass(frozen=True)
class QVARResult:
    """Quantile impulse-response estimates for one directed pair."""

    node_i: str
    node_j: str
    feature_col: str
    tau: float
    # β_cross: quantile impulse-response of j to a unit shock in i
    beta_cross: float
    # β_own: autoregressive component of j
    beta_own: float
    # Pseudo-R² from QuantReg
    pseudo_r2: float
    # t-statistic and p-value for β_cross
    t_stat: float
    p_value: float
    n_obs: int
    converged: bool


def run_qvar(
    series_i: np.ndarray,
    series_j: np.ndarray,
    node_i: str = "node_i",
    node_j: str = "node_j",
    feature_col: str = "",
    quantiles: tuple[float, ...] = _DEFAULT_QUANTILES,
    lags: int = _DEFAULT_LAGS,
) -> list[QVARResult]:
    """Estimate pairwise QVAR for the pair (i→j) at each quantile in *quantiles*.

    Args:
        series_i:   Source series (potential contagion driver), T observations.
        series_j:   Response series, T observations.
        node_i:     Label for source node.
        node_j:     Label for response node.
        feature_col: Feature name for traceability.
        quantiles:  Quantiles at which to estimate (default 0.05, 0.50, 0.95).
        lags:       Number of lags (default 1).

    Returns:
        List of QVARResult, one per quantile.
    """
    if not _HAS_STATSMODELS:
        raise ImportError(
            "statsmodels is required for QVAR. Install with: pip install statsmodels"
        )

    xi = np.asarray(series_i, dtype=float)
    xj = np.asarray(series_j, dtype=float)

    # Build lagged design matrix
    T = min(len(xi), len(xj))
    xi = xi[:T]
    xj = xj[:T]

    n = T - lags
    if n < 20:
        return [
            QVARResult(
                node_i=node_i, node_j=node_j, feature_col=feature_col,
                tau=tau, beta_cross=float("nan"), beta_own=float("nan"),
                pseudo_r2=float("nan"), t_stat=float("nan"), p_value=float("nan"),
                n_obs=n, converged=False,
            )
            for tau in quantiles
        ]

    y  = xj[lags:]
    xi_lag = xi[lags - 1 : T - 1] if lags == 1 else xi[: T - lags]
    xj_lag = xj[lags - 1 : T - 1] if lags == 1 else xj[: T - lags]

    df = pd.DataFrame({"y": y, "xi_lag": xi_lag, "xj_lag": xj_lag})

    results = []
    for tau in quantiles:
        try:
            mod = smf.quantreg("y ~ xi_lag + xj_lag", data=df)
            fit = mod.fit(q=tau, max_iter=2000)
            params = fit.params
            pvalues = fit.pvalues
            tvalues = fit.tvalues
            beta_cross = float(params.get("xi_lag", float("nan")))
            beta_own   = float(params.get("xj_lag", float("nan")))
            t_stat     = float(tvalues.get("xi_lag", float("nan")))
            p_value    = float(pvalues.get("xi_lag", float("nan")))
            pseudo_r2  = float(getattr(fit, "prsquared", float("nan")))
            converged  = True
        except Exception:
            beta_cross = float("nan")
            beta_own   = float("nan")
            t_stat     = float("nan")
            p_value    = float("nan")
            pseudo_r2  = float("nan")
            converged  = False

        results.append(QVARResult(
            node_i=node_i, node_j=node_j, feature_col=feature_col,
            tau=tau, beta_cross=beta_cross, beta_own=beta_own,
            pseudo_r2=pseudo_r2, t_stat=t_stat, p_value=p_value,
            n_obs=n, converged=converged,
        ))

    return results
