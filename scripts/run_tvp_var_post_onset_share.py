"""Post-onset share of TVP-VAR cross-edge FEVD (makes the 'spillover is post-onset' claim reproducible).

The paper states the rolling TVP-VAR places the spillover post-onset with near-zero pre-onset. This
computes that directly from the committed rolling-spillovers tables: for each event, sum the off-diagonal
(cross-node) FEVD share over post-onset windows (window_center >= 0) and divide by the total over all
windows. Reports per-event and pooled. No new data; pure recompute of committed artifacts.

Reads:  results/tables/table_tvp_var_spillovers_{event}.csv
Output: results/tables/tvp_var_post_onset_share.json
"""
from __future__ import annotations
import csv, glob, json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPILL = ROOT / "results" / "tables"


def event_share(rows):
    cross = [r for r in rows if r["causing_node"] != r["caused_node"]]
    tot = sum(float(r["fevd_share"]) for r in cross)
    post = sum(float(r["fevd_share"]) for r in cross if float(r["window_center"]) >= 0)
    pre = tot - post
    nwin = len(set(float(r["window_center"]) for r in rows))
    return tot, post, pre, nwin


def main():
    per_event = {}
    pool_post = pool_tot = 0.0
    for f in sorted(glob.glob(str(SPILL / "table_tvp_var_spillovers_*.csv"))):
        rows = list(csv.DictReader(open(f)))
        ev = rows[0]["event_id"]
        tot, post, pre, nwin = event_share(rows)
        pool_post += post; pool_tot += tot
        per_event[ev] = {"post_onset_fevd_share": round(post / tot, 4) if tot else None,
                         "pre_onset_fevd_share": round(pre / tot, 4) if tot else None,
                         "n_windows": nwin}
    out = {"metric": ("Off-diagonal (cross-node) FEVD share occurring in post-onset rolling windows "
                      "(window_center >= 0), from the committed TVP-VAR rolling spillovers."),
           "per_event": per_event,
           "pooled_post_onset_share": round(pool_post / pool_tot, 4),
           "per_event_min": round(min(v["post_onset_fevd_share"] for v in per_event.values()), 4),
           "per_event_max": round(max(v["post_onset_fevd_share"] for v in per_event.values()), 4)}
    (SPILL / "tvp_var_post_onset_share.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
