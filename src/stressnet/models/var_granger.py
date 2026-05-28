"""VAR / Granger causality and FEVD spillover estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from stressnet.models.leadlag import fdr_correct
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

    n_obs, n_eq = data.shape
    max_estimable_lags = max(1, (n_obs - 1) // (n_eq + 1) - 1)
    if max_lags > max_estimable_lags:
        logger.warning(
            "Clamping VAR max_lags from %d to %d for %d observations × %d equations.",
            max_lags,
            max_estimable_lags,
            n_obs,
            n_eq,
        )
        max_lags = max_estimable_lags

    # Wrap as pandas DataFrame so statsmodels stores proper column names;
    # otherwise test_causality receives generic 'y1'/'y2'/... names that
    # don't match our node_names strings.
    df_pd = pd.DataFrame(data, columns=node_names)
    model = VAR(df_pd)
    try:
        results = model.fit(maxlags=max_lags, ic=ic)
    except Exception as exc:
        logger.warning(
            "VAR lag selection failed (%s). Falling back to fixed lag order search.",
            exc,
        )
        last_exc: Exception | None = None
        for lag_order in range(max_lags, 0, -1):
            try:
                results = model.fit(lag_order)
                break
            except Exception as candidate_exc:
                last_exc = candidate_exc
        else:
            raise last_exc or exc
    try:
        logger.info("VAR fitted: lag order %d, AIC %.4f", results.k_ar, results.aic)
    except Exception:
        logger.info("VAR fitted: lag order %d; information criteria unavailable.", results.k_ar)
    return {"model": model, "results": results, "lag_order": results.k_ar, "node_names": node_names}


def granger_causality_table(
    var_fit: dict,
    significance_level: float = 0.05,
) -> pl.DataFrame:
    """Compute pairwise Granger causality test results from a fitted VAR.

    Columns returned:
        causing_node, caused_node, f_stat, p_value,
        significant_p05      (unadjusted Bonferroni-like threshold for reporting)
        p_value_fdr          (BH-FDR adjusted p-value)
        significant_fdr      (FDR < significance_level)
        p_bonferroni         (Bonferroni adjusted p-value)
        significant_bonferroni
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
                })
            except Exception as exc:
                logger.debug("Granger test failed for %s→%s: %s", causing, caused, exc)

    _empty_schema = {
        "causing_node": pl.String, "caused_node": pl.String,
        "f_stat": pl.Float64, "p_value": pl.Float64,
        "significant_p05": pl.Boolean,
        "p_value_fdr": pl.Float64, "significant_fdr": pl.Boolean,
        "p_bonferroni": pl.Float64, "significant_bonferroni": pl.Boolean,
    }
    if not rows:
        return pl.DataFrame(schema=_empty_schema)

    df = pl.DataFrame(rows)
    n_tests = len(df)
    p_arr = df["p_value"].to_numpy()

    # BH-FDR
    reject_fdr, adj_p_fdr = fdr_correct(p_arr, alpha=significance_level)

    # Bonferroni
    p_bonf = np.minimum(p_arr * n_tests, 1.0)
    sig_bonf = p_bonf < significance_level

    return df.with_columns(
        (pl.col("p_value") < significance_level).alias("significant_p05"),
        pl.Series("p_value_fdr",          adj_p_fdr),
        pl.Series("significant_fdr",      reject_fdr),
        pl.Series("p_bonferroni",         p_bonf),
        pl.Series("significant_bonferroni", sig_bonf),
    )


def fevd_spillover_table(var_fit: dict, horizon: int = 10) -> pl.DataFrame:
    """Compute Forecast-Error Variance Decomposition spillover shares.

    Returns a DataFrame with columns: caused_node, causing_node, fevd_share.
    """
    results = var_fit["results"]
    names = var_fit["node_names"]
    try:
        fevd = results.fevd(periods=horizon).decomp  # shape (T, N, N)
    except Exception as exc:
        logger.warning(
            "FEVD failed (%s). Using normalized absolute VAR coefficients as a fallback "
            "spillover proxy.",
            exc,
        )
        coefs = np.abs(results.coefs).sum(axis=0)
        row_sums = coefs.sum(axis=1, keepdims=True) + 1e-12
        shares = coefs / row_sums
        rows = []
        for i, caused in enumerate(names):
            for j, causing in enumerate(names):
                rows.append({
                    "caused_node": caused,
                    "causing_node": causing,
                    "fevd_share": float(shares[i, j]),
                    "horizon": horizon,
                    "method": "var_coeff_fallback",
                })
        return pl.DataFrame(rows)

    rows = []
    h_idx = min(horizon - 1, len(fevd) - 1)  # clamp to available periods
    fevd_at_h = fevd[h_idx]  # (N, N) at the specified horizon
    for i, caused in enumerate(names):
        for j, causing in enumerate(names):
            rows.append({
                "caused_node": caused,
                "causing_node": causing,
                "fevd_share": float(fevd_at_h[i, j]),
                "horizon": horizon,
                "method": "fevd",
            })
    return pl.DataFrame(rows)
