"""Transfer entropy estimation for non-linear directional information flow."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import entropy as scipy_entropy

from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def _conditional_entropy_binned(
    y_future: np.ndarray,
    y_past: np.ndarray,
    x_past: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Estimate H(Y_t | Y_{t-1}, X_{t-1}) - H(Y_t | Y_{t-1}) via binning.

    This is the discrete approximation of transfer entropy.
    """
    # Bin all signals
    y_f = np.digitize(y_future, np.percentile(y_future, np.linspace(0, 100, n_bins + 1)[1:-1]))
    y_p = np.digitize(y_past, np.percentile(y_past, np.linspace(0, 100, n_bins + 1)[1:-1]))
    x_p = np.digitize(x_past, np.percentile(x_past, np.linspace(0, 100, n_bins + 1)[1:-1]))

    n = len(y_f)
    n_bins_actual = n_bins - 1  # after digitize

    # H(Y_t | Y_{t-1}) = H(Y_t, Y_{t-1}) - H(Y_{t-1})
    joint_yy = np.zeros((n_bins_actual, n_bins_actual))
    joint_yyx = np.zeros((n_bins_actual, n_bins_actual, n_bins_actual))

    for t in range(n):
        yf, yp_, xp_ = min(y_f[t], n_bins_actual - 1), min(y_p[t], n_bins_actual - 1), min(x_p[t], n_bins_actual - 1)
        joint_yy[yf, yp_] += 1
        joint_yyx[yf, yp_, xp_] += 1

    joint_yy /= (n + 1e-12)
    joint_yyx /= (n + 1e-12)

    # H(Y | Y_past) via marginalisation
    p_ypast = joint_yy.sum(axis=0)
    p_yfut_given_ypast = joint_yy / (p_ypast + 1e-12)
    h_y_given_ypast = -np.sum(joint_yy * np.log(p_yfut_given_ypast + 1e-12))

    # H(Y | Y_past, X_past)
    p_yxpast = joint_yyx.sum(axis=0)
    p_yfut_given_yxpast = joint_yyx / (p_yxpast + 1e-12)
    h_y_given_yxpast = -np.sum(joint_yyx * np.log(p_yfut_given_yxpast + 1e-12))

    te = h_y_given_ypast - h_y_given_yxpast
    return max(0.0, float(te))


def transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    history_length: int = 10,
    n_bins: int = 10,
) -> float:
    """Estimate TE(X→Y): how much X's past reduces uncertainty about Y's future.

    Uses binned discrete approximation. For more accurate estimates with
    continuous data, prefer a KNN estimator (not included in base dependencies).
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
    """Compute a null TE distribution by shuffling x's time series."""
    if rng is None:
        rng = np.random.default_rng(42)
    null_tes = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        x_shuffled = rng.permutation(x)
        null_tes[i] = transfer_entropy(x_shuffled, y, history_length, n_bins)
    return null_tes


def compute_te_table(
    panel: pl.DataFrame,
    node_pairs: list[tuple[str, str]],
    feature_col: str = "basis_vs_usd",
    history_length: int = 10,
    n_bins: int = 10,
    n_shuffles: int = 200,
    ts_col: str = "event_time_seconds",
) -> pl.DataFrame:
    """Compute pairwise transfer entropy and p-values.

    Returns a DataFrame suitable for export as table_transfer_entropy.csv.
    """
    rng = np.random.default_rng(42)
    results = []

    for node_i, node_j in node_pairs:
        xi = panel.filter(pl.col("node_id") == node_i).sort(ts_col)[feature_col].drop_nulls().to_numpy()
        xj = panel.filter(pl.col("node_id") == node_j).sort(ts_col)[feature_col].drop_nulls().to_numpy()
        if len(xi) < 30 or len(xj) < 30:
            continue

        te_val = transfer_entropy(xi, xj, history_length, n_bins)
        null_dist = te_null_distribution(xi, xj, history_length, n_bins, n_shuffles, rng)
        p_val = float(np.mean(null_dist >= te_val))

        results.append({
            "node_i": node_i,
            "node_j": node_j,
            "te_i_to_j": te_val,
            "p_value": p_val,
            "significant_p05": p_val < 0.05,
        })

    return pl.DataFrame(results).sort("te_i_to_j", descending=True)
