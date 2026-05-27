"""VAR / Granger causality and FEVD spillover estimation."""

from __future__ import annotations

import numpy as np
import polars as pl

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from statsmodels.tsa.api import VAR
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    logger.warning("statsmodels not available; VAR/Granger estimation will fail.")


def fit_var(
    data: np.ndarray,
    node_names: list[str],
    max_lags: int = 10,
    ic: str = "bic",
) -> dict:
    """Fit a Vector Autoregression model and return results dict.

    Args:
        data: (T, N) array of node-level time series.
        node_names: List of node names (length N).
        max_lags: Maximum lag order to test.
        ic: Information criterion for lag selection ('bic', 'aic', 'hqic').

    Returns:
        Dict with keys: model, results, lag_order, node_names.
    """
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels is required for VAR estimation.")

    model = VAR(data)
    results = model.fit(maxlags=max_lags, ic=ic)
    logger.info("VAR fitted: lag order %d, AIC %.4f", results.k_ar, results.aic)
    return {"model": model, "results": results, "lag_order": results.k_ar, "node_names": node_names}


def granger_causality_table(
    var_fit: dict,
    significance_level: float = 0.05,
) -> pl.DataFrame:
    """Compute pairwise Granger causality test results from a fitted VAR.

    Returns a DataFrame with columns: causing_node, caused_node, f_stat, p_value, significant.
    """
    results = var_fit["results"]
    names = var_fit["node_names"]
    rows = []
    for caused in names:
        for causing in names:
            if causing == caused:
                continue
            try:
                test = results.test_causality(caused, causing, kind="f")
                rows.append({
                    "causing_node": causing,
                    "caused_node": caused,
                    "f_stat": float(test.test_statistic),
                    "p_value": float(test.pvalue),
                    "significant": float(test.pvalue) < significance_level,
                })
            except Exception as exc:
                logger.debug("Granger test failed for %s→%s: %s", causing, caused, exc)
    return pl.DataFrame(rows)


def fevd_spillover_table(var_fit: dict, horizon: int = 10) -> pl.DataFrame:
    """Compute Forecast-Error Variance Decomposition spillover shares.

    Returns a DataFrame with columns: caused_node, causing_node, fevd_share.
    """
    results = var_fit["results"]
    names = var_fit["node_names"]
    fevd = results.fevd(periods=horizon).decomp  # shape (T, N, N)

    rows = []
    fevd_at_h = fevd[horizon - 1]  # (N, N) at the specified horizon
    for i, caused in enumerate(names):
        for j, causing in enumerate(names):
            rows.append({
                "caused_node": caused,
                "causing_node": causing,
                "fevd_share": float(fevd_at_h[i, j]),
                "horizon": horizon,
            })
    return pl.DataFrame(rows)
