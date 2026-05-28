"""Transfer entropy estimation for non-linear directional information flow."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import entropy as scipy_entropy

from stressnet.models.leadlag import fdr_correct
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def _conditional_entropy_binned(
    y_future: np.ndarray,
    y_past: np.ndarray,
    x_past: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Estimate H(Y_t | Y_{t-1}, X_{t-1}) - H(Y_t | Y_{t-1}) via binning."""
    y_f = np.digitize(y_future, np.percentile(y_future, np.linspace(0, 100, n_bins + 1)[1:-1]))
    y_p = np.digitize(y_past, np.percentile(y_past, np.linspace(0, 100, n_bins + 1)[1:-1]))
    x_p = np.digitize(x_past, np.percentile(x_past, np.linspace(0, 100, n_bins + 1)[1:-1]))

    n = len(y_f)
    nb = n_bins - 1  # effective bin count after digitize

    joint_yy = np.zeros((nb, nb))
    joint_yyx = np.zeros((nb, nb, nb))

    for t in range(n):
        yf = min(y_f[t], nb - 1)
        yp_ = min(y_p[t], nb - 1)
        xp_ = min(x_p[t], nb - 1)
        joint_yy[yf, yp_] += 1
        joint_yyx[yf, yp_, xp_] += 1

    joint_yy /= (n + 1e-12)
    joint_yyx /= (n + 1e-12)

    p_ypast = joint_yy.sum(axis=0)
    p_yfut_given_ypast = joint_yy / (p_ypast + 1e-12)
    h_y_given_ypast = -np.sum(joint_yy * np.log(p_yfut_given_ypast + 1e-12))

    p_yxpast = joint_yyx.sum(axis=0)
    p_yfut_given_yxpast = joint_yyx / (p_yxpast + 1e-12)
    h_y_given_yxpast = -np.sum(joint_yyx * np.log(p_yfut_given_yxpast + 1e-12))

    return max(0.0, float(h_y_given_ypast - h_y_given_yxpast))


def transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    history_length: int = 10,
    n_bins: int = 10,
) -> float:
    """Estimate TE(X→Y): how much X's past reduces uncertainty about Y's future.

    Uses binned discrete approximation.
    """
    x_clean = x[~np.isnan(x)]
    y_clean = y[~np.isnan(y)]
    n = min(len(x_clean), len(y_clean)) - history_length
    if n < 20:
        return 0.0

    y_future = y_clean[history_length:]
    y_past = y_clean[:-history_length]
    x_past = x_clean[:-history_length]

    return _conditional_entropy_binned(y_future[:n], y_past[:n], x_past[:n], n_bins=n_bins)


def te_null_distribution(
    x: np.ndarray,
    y: np.ndarray,
    history_length: int = 10,
    n_bins: int = 10,
    n_shuffles: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute a null TE distribution by randomly permuting x's time series.

    Simple permutation destroys all temporal dependence; good as a lower bound
    on the null but can be anti-conservative for auto-correlated series.
    Use te_block_null_distribution for a more conservative block-shuffle null.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    null_tes = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        x_shuffled = rng.permutation(x)
        null_tes[i] = transfer_entropy(x_shuffled, y, history_length, n_bins)
    return null_tes


def te_block_null_distribution(
    x: np.ndarray,
    y: np.ndarray,
    history_length: int = 10,
    n_bins: int = 10,
    n_shuffles: int = 200,
    block_size: int = 60,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Block-shuffle null: permute contiguous blocks of x, preserving intra-block
    autocorrelation.  This is more conservative (larger p-values) than the simple
    permutation null and better controls false positives for persistent series.

    Args:
        block_size: Number of observations per block. Default 60 = one hour of
                    1-minute data, which preserves short-term autocorrelation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    n_blocks = max(1, n // block_size)
    null_tes = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        block_order = rng.permutation(n_blocks)
        blocks = [x[b * block_size: (b + 1) * block_size] for b in block_order]
        x_shuffled = np.concatenate(blocks)[:n]
        null_tes[i] = transfer_entropy(x_shuffled, y, history_length, n_bins)
    return null_tes


def compute_te_table(
    panel: pl.DataFrame,
    node_pairs: list[tuple[str, str]],
    feature_col: str = "basis_vs_usd",
    history_length: int = 10,
    n_bins: int = 10,
    n_shuffles: int = 200,
    block_size: int = 60,
    ts_col: str = "event_time_seconds",
    fdr_alpha: float = 0.05,
) -> pl.DataFrame:
    """Compute pairwise TE with both iid-shuffle and block-shuffle nulls.

    Columns returned:
        node_i, node_j, te_i_to_j,
        p_value              (iid-shuffle null)
        p_value_block        (block-shuffle null — more conservative)
        significant_p05      (iid p < 0.05)
        p_value_fdr          (BH-FDR on iid p-values)
        significant_fdr
        p_value_block_fdr    (BH-FDR on block p-values)
        significant_block_fdr
        p_bonferroni         (Bonferroni on iid p-values)
        significant_bonferroni
    """
    rng = np.random.default_rng(42)
    results = []

    for node_i, node_j in node_pairs:
        xi = (
            panel.filter(pl.col("node_id") == node_i)
            .sort(ts_col)[feature_col]
            .drop_nulls()
            .to_numpy()
        )
        xj = (
            panel.filter(pl.col("node_id") == node_j)
            .sort(ts_col)[feature_col]
            .drop_nulls()
            .to_numpy()
        )
        if len(xi) < 30 or len(xj) < 30:
            continue

        te_val = transfer_entropy(xi, xj, history_length, n_bins)
        null_iid   = te_null_distribution(xi, xj, history_length, n_bins, n_shuffles, rng)
        null_block = te_block_null_distribution(xi, xj, history_length, n_bins, n_shuffles,
                                                block_size=block_size, rng=rng)
        p_iid   = float(np.mean(null_iid   >= te_val))
        p_block = float(np.mean(null_block >= te_val))

        results.append({
            "node_i": node_i,
            "node_j": node_j,
            "te_i_to_j": te_val,
            "p_value": p_iid,
            "p_value_block": p_block,
            "significant_p05": p_iid < 0.05,
        })

    if not results:
        return pl.DataFrame(schema={
            "node_i": pl.String, "node_j": pl.String,
            "te_i_to_j": pl.Float64,
            "p_value": pl.Float64, "p_value_block": pl.Float64,
            "significant_p05": pl.Boolean,
            "p_value_fdr": pl.Float64, "significant_fdr": pl.Boolean,
            "p_value_block_fdr": pl.Float64, "significant_block_fdr": pl.Boolean,
            "p_bonferroni": pl.Float64, "significant_bonferroni": pl.Boolean,
        })

    df = pl.DataFrame(results).sort("te_i_to_j", descending=True)
    n_tests = len(df)

    # BH-FDR on iid p-values
    p_arr_iid = df["p_value"].to_numpy()
    reject_fdr, adj_p_fdr = fdr_correct(p_arr_iid, alpha=fdr_alpha)

    # BH-FDR on block p-values
    p_arr_block = df["p_value_block"].to_numpy()
    reject_block_fdr, adj_p_block_fdr = fdr_correct(p_arr_block, alpha=fdr_alpha)

    # Bonferroni on iid p-values
    p_bonferroni = np.minimum(p_arr_iid * n_tests, 1.0)
    sig_bonferroni = p_bonferroni < fdr_alpha

    return df.with_columns(
        pl.Series("p_value_fdr",            adj_p_fdr),
        pl.Series("significant_fdr",        reject_fdr),
        pl.Series("p_value_block_fdr",      adj_p_block_fdr),
        pl.Series("significant_block_fdr",  reject_block_fdr),
        pl.Series("p_bonferroni",           p_bonferroni),
        pl.Series("significant_bonferroni", sig_bonferroni),
    )
