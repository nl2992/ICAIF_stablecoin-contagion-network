"""Label-shuffle permutation null for the USDT/Curve regime-contagion shift (24_run_regime_contagion.py).

The block bootstrap tests sampling variability; this tests whether the calm/panic SPLIT itself is
meaningful — directly addressing the small acute window (n=53). We pool the hourly cross-pool flow pairs
from both phases, hold the pre/panic group sizes fixed, randomly reassign which hours are 'panic'
(1000x), and recompute the Fisher r-to-z. The permutation p-value = fraction of shuffles whose z is at
least the observed z. We first REPRODUCE the committed numbers (rho_pre 0.09, rho_panic 0.53, z 2.82);
if they do not match, abort (mis-specified setup).

Output -> results/tables/regime_permutation_null.json
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stressnet.config import gold_root  # noqa: E402

EVENT, A, B, FEATURE = "usdt_curve_2023", "curve_3pool", "curve_crvusd_usdt", "usdc_net_sold_1h"


def hourly(sub, node):
    return (sub.filter(pl.col("node_id") == node)
            .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
            .group_by("h").agg(pl.col(FEATURE).sum().alias("f")))


def phase_pairs(df, phase):
    sub = df.filter((pl.col("event_phase") == phase) & pl.col(FEATURE).is_not_null())
    m = hourly(sub, A).rename({"f": "a"}).join(hourly(sub, B).rename({"f": "b"}), on="h", how="inner").drop_nulls()
    return m["a"].to_numpy(), m["b"].to_numpy()


def fisher_z(a1, b1, a2, b2):
    if np.std(a1) == 0 or np.std(b1) == 0 or np.std(a2) == 0 or np.std(b2) == 0:
        return None
    r1 = np.clip(np.corrcoef(a1, b1)[0, 1], -0.999, 0.999)
    r2 = np.clip(np.corrcoef(a2, b2)[0, 1], -0.999, 0.999)
    n1, n2 = len(a1), len(a2)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    return (math.atanh(r2) - math.atanh(r1)) / se, float(r1), float(r2)


def main(n_perm=2000, seed=0):
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{EVENT}.parquet")
    a_pre, b_pre = phase_pairs(df, "pre")
    a_pan, b_pan = phase_pairs(df, "panic")
    z_obs, r_pre, r_pan = fisher_z(a_pre, b_pre, a_pan, b_pan)
    n_pre, n_pan = len(a_pre), len(a_pan)
    print(f"OBSERVED: rho_pre={r_pre:.4f} (n={n_pre}), rho_panic={r_pan:.4f} (n={n_pan}), z={z_obs:.4f}")
    # safety check vs committed (rho_pre 0.0905, rho_panic 0.5273, z 2.8209)
    assert abs(r_pre - 0.0905) < 0.01 and abs(r_pan - 0.5273) < 0.01 and abs(z_obs - 2.8209) < 0.02, \
        "Reproduction mismatch — setup differs from committed; aborting."

    a_all = np.concatenate([a_pre, a_pan]); b_all = np.concatenate([b_pre, b_pan])
    N = len(a_all); rng = np.random.default_rng(seed)
    ge = 0; valid = 0
    for _ in range(n_perm):
        idx = rng.permutation(N)
        pan = idx[:n_pan]; pre = idx[n_pan:]
        res = fisher_z(a_all[pre], b_all[pre], a_all[pan], b_all[pan])
        if res is None:
            continue
        valid += 1
        if res[0] >= z_obs:
            ge += 1
    p_perm = ge / valid
    out = {"event": EVENT, "pair": [A, B], "feature": FEATURE,
           "rho_pre": round(r_pre, 4), "n_pre": n_pre, "rho_panic": round(r_pan, 4), "n_panic": n_pan,
           "fisher_z_obs": round(z_obs, 4), "n_permutations": valid, "n_z_ge_obs": ge,
           "p_permutation_one_sided": round(p_perm, 4),
           "interpretation": ("One-sided label-shuffle null on the pre/panic split: fraction of random "
                              "95/53 re-labellings whose Fisher z >= the observed. Small p => the linkage "
                              "rise is tied to the actual onset timing, not the small acute window.")}
    (ROOT / "results/tables/regime_permutation_null.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
