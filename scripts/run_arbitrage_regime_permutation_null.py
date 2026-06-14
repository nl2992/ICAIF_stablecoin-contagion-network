"""Label-shuffle permutation null for the arbitrage stabilizing->amplifying flip (25_run_arbitrage_regime.py).

The flip is reported with an analytic Fisher two-sample p (parametric, normality-dependent, small panic
window). This is the non-parametric counterpart, identical in spirit to run_regime_permutation_null.py for
the contagion coupling: pool the hourly (|flow|,|price-dev|) pairs from both regimes, hold the calm/panic
split sizes fixed, randomly reassign which hours are 'panic' (B=2000), and recompute the Fisher z. The
permutation p = fraction of relabellings whose |z| >= the observed |z| (two-sided, matching p_shift). We
reproduce the committed z first as a gate (abort on mismatch).

Output -> results/tables/arbitrage_regime_permutation_null.json
"""
from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stressnet.config import gold_root  # noqa: E402

_spec = importlib.util.spec_from_file_location("arb", ROOT / "scripts" / "25_run_arbitrage_regime.py")
arb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(arb)

# committed z (table_arbitrage_regime.csv) — the three events cited in the paper
GATE = {"usdt_curve_2023": 3.8364, "busd_2023": 3.6931, "ftx_2022": -2.8027}


def phase_pairs(df, phase, cex):
    sub = df.filter(pl.col("event_phase") == phase)
    A = arb._hourly_abs(sub, arb._FLOW_NODE, arb._FLOW_COL).rename({"v": "a"})
    B = arb._hourly_abs(sub, cex, arb._PRICE_COL).rename({"v": "b"})
    m = A.join(B, on="h", how="inner").drop_nulls().sort("h")  # deterministic row order
    return np.abs(m["a"].to_numpy()), np.abs(m["b"].to_numpy())


def fisher_z(x1, y1, x2, y2):
    if np.std(x1) == 0 or np.std(y1) == 0 or np.std(x2) == 0 or np.std(y2) == 0:
        return None
    c = lambda r: max(min(r, 0.999), -0.999)
    r1 = c(stats.pearsonr(x1, y1)[0]); r2 = c(stats.pearsonr(x2, y2)[0])
    n1, n2 = len(x1), len(x2)
    return (math.atanh(r2) - math.atanh(r1)) / math.sqrt(1/(n1-3) + 1/(n2-3)), r1, r2


def run_event(ev, cex, n_perm=10000, seed=0):
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{ev}.parquet")
    xc, yc = phase_pairs(df, "pre", cex)
    xp, yp = phase_pairs(df, "panic", cex)
    z_obs, r_c, r_p = fisher_z(xc, yc, xp, yp)
    n_c, n_p = len(xc), len(xp)
    gate = GATE[ev]
    assert abs(round(z_obs, 4) - gate) < 0.02, f"{ev}: z {z_obs:.4f} != committed {gate}; aborting."
    x_all = np.concatenate([xc, xp]); y_all = np.concatenate([yc, yp])
    N = len(x_all); rng = np.random.default_rng(seed)
    ge = 0; valid = 0
    for _ in range(n_perm):
        idx = rng.permutation(N)
        pan = idx[:n_p]; pre = idx[n_p:]
        res = fisher_z(x_all[pre], y_all[pre], x_all[pan], y_all[pan])
        if res is None:
            continue
        valid += 1
        if abs(res[0]) >= abs(z_obs):
            ge += 1
    return {"event": ev, "cex_node": cex, "r_calm": round(r_c, 4), "n_calm": n_c,
            "r_panic": round(r_p, 4), "n_panic": n_p, "fisher_z_obs": round(z_obs, 4),
            "flip_sign": bool(r_c < 0 and r_p > 0), "n_permutations": valid, "n_absz_ge_obs": ge,
            "p_permutation_two_sided": round(ge / valid, 4)}


def main():
    out = {"method": ("Two-sided label-shuffle null on the calm/panic split of the on-chain-flow vs "
                      "CEX-price-deviation correlation; p = fraction of relabellings with |Fisher z| >= "
                      "observed. Non-parametric counterpart to the analytic p_shift."), "events": {}}
    for ev, cex in [("usdt_curve_2023", "usdt_binance"), ("busd_2023", "busd_binance"),
                    ("ftx_2022", "usdt_binance")]:
        r = run_event(ev, cex)
        out["events"][ev] = r
        print(f"{ev}: z_obs={r['fisher_z_obs']} flip={r['flip_sign']} "
              f"n_calm={r['n_calm']} n_panic={r['n_panic']} p_perm={r['p_permutation_two_sided']}")
    (ROOT / "results/tables/arbitrage_regime_permutation_null.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["events"], indent=2))


if __name__ == "__main__":
    main()
