"""Plan H — Signal taxonomy table as the paper's conceptual anchor.

Synthesises existing results (Forbes-Rigobon, HMM, lead-lag, online detection)
into a 2×2 taxonomy of when on-chain AMM signals work:

  Rows: shock_type   — endogenous (pool-level imbalance) vs exogenous (market shock)
  Cols: signal_layer — on-chain AMM vs CEX market

Each cell: detected / not detected, which method wins, detection lead in hours.

Outputs both a LaTeX table (for direct inclusion in the paper) and a CSV.

Reads:  results/tables/table_regime_contagion.csv
        results/tables/table_hmm_regime.csv
        results/tables/table_online_detection.csv
        results/tables/table_arbitrage_regime.csv   (if present)
Writes: results/tables/table_signal_taxonomy.csv
        results/figures/fig_signal_taxonomy.tex

Usage:
    python scripts/make_taxonomy_figure.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import warnings

import polars as pl

from stressnet.config import results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

# Event metadata: (shock_type, mechanism_label)
_EVENT_META = {
    "usdt_curve_2023":  ("endogenous",  "DEX pool imbalance"),
    "terra_luna_2022":  ("endogenous",  "Algorithmic reflexive"),
    "ftx_2022":         ("exogenous",   "Exchange credit shock"),
    "busd_2023":        ("exogenous",   "Regulatory issuer wind-down"),
    "usdc_svb_2023":    ("exogenous",   "Fiat reserve bank shock"),
}


def _read_csv_as_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(v) -> float | None:
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def _safe_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def _latex_cell(detected: bool, method: str, lead: str | None) -> str:
    if detected:
        lead_str = f"(+{lead}h)" if lead else ""
        return r"\cellcolor{green!15}\textbf{" + method + r"} " + lead_str
    return r"\cellcolor{red!10}not detected"


def main() -> None:
    tdir = results_root() / "tables"
    fdir = results_root() / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    fr_rows   = {r["event_id"]: r for r in _read_csv_as_dicts(tdir / "table_regime_contagion.csv")}
    hmm_rows  = {r["event_id"]: r for r in _read_csv_as_dicts(tdir / "table_hmm_regime.csv")}
    det_rows  = {r["event"]: r    for r in _read_csv_as_dicts(tdir / "table_online_detection.csv")}
    # Prefer the new detailed intensity table; fall back to script 25 regime table
    _arb_intensity = _read_csv_as_dicts(tdir / "table_arbitrage_intensity.csv")
    _arb_regime    = _read_csv_as_dicts(tdir / "table_arbitrage_regime.csv")
    arb_rows = {r["event_id"]: r for r in _arb_intensity} or \
               {r["event_id"]: r for r in _arb_regime}

    taxonomy_rows = []
    for event_id, (shock_type, mechanism) in _EVENT_META.items():
        fr  = fr_rows.get(event_id, {})
        hmm = hmm_rows.get(event_id, {})
        det = det_rows.get(event_id, {})
        arb = arb_rows.get(event_id, {})

        # On-chain signal layer
        onchain_detected_fr  = _safe_bool(fr.get("contagion_regime_shift", False))
        onchain_detected_hmm = _safe_bool(hmm.get("detects_regime", False))
        onchain_auroc        = _safe_float(hmm.get("auroc"))
        onchain_fisher_z     = _safe_float(fr.get("fisher_z"))
        onchain_lead_h       = _safe_float(det.get("delay_onchain_h"))  # negative = leads

        # CEX market signal layer
        market_auroc   = _safe_float(det.get("auroc_market_causal"))
        market_lead_h  = _safe_float(det.get("delay_market_h"))    # negative = leads

        # Earlier-by (from online detection)
        onchain_earlier_h = _safe_float(det.get("earlier_by_h"))

        # Determine winner
        if onchain_lead_h is not None and market_lead_h is not None:
            if onchain_lead_h < market_lead_h:  # more negative = fires earlier
                winner = "on-chain AMM"
            elif market_lead_h < onchain_lead_h:
                winner = "CEX market"
            else:
                winner = "tie"
        else:
            winner = "n/a"

        arb_regime_flip = _safe_bool(arb.get("arb_regime_flip", False))

        taxonomy_rows.append({
            "event_id":                event_id,
            "shock_type":              shock_type,
            "mechanism":               mechanism,
            "onchain_fr_detected":     onchain_detected_fr,
            "onchain_hmm_detected":    onchain_detected_hmm,
            "onchain_hmm_auroc":       round(onchain_auroc, 3) if onchain_auroc else None,
            "onchain_fisher_z":        round(onchain_fisher_z, 3) if onchain_fisher_z else None,
            "onchain_lead_h":          int(onchain_lead_h) if onchain_lead_h else None,
            "market_auroc":            round(market_auroc, 3) if market_auroc else None,
            "market_lead_h":           int(market_lead_h) if market_lead_h else None,
            "onchain_earlier_by_h":    int(onchain_earlier_h) if onchain_earlier_h else None,
            "signal_winner":           winner,
            "arb_regime_flip":         arb_regime_flip,
        })

    # Write CSV
    csv_path = tdir / "table_signal_taxonomy.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(taxonomy_rows[0].keys()))
        w.writeheader(); w.writerows(taxonomy_rows)

    # Build LaTeX table
    endo  = [r for r in taxonomy_rows if r["shock_type"] == "endogenous"]
    exo   = [r for r in taxonomy_rows if r["shock_type"] == "exogenous"]

    def _fmt_row(r: dict) -> str:
        oc_det  = "\\checkmark" if r["onchain_hmm_detected"] else "---"
        oc_auc  = f"{r['onchain_hmm_auroc']:.3f}" if r["onchain_hmm_auroc"] else "---"
        oc_z    = f"{r['onchain_fisher_z']:+.2f}" if r["onchain_fisher_z"] else "---"
        oc_lead = f"{r['onchain_lead_h']:+d}h" if r["onchain_lead_h"] is not None else "---"
        mk_auc  = f"{r['market_auroc']:.3f}" if r["market_auroc"] else "---"
        mk_lead = f"{r['market_lead_h']:+d}h" if r["market_lead_h"] is not None else "---"
        adv_h   = f"{r['onchain_earlier_by_h']:+d}h" if r["onchain_earlier_by_h"] is not None else "---"
        flip    = "yes" if r["arb_regime_flip"] else "no"
        event_label = r["event_id"].replace("_", r"\_")
        mech    = r["mechanism"].replace("&", r"\&")
        return (
            f"    {event_label} & {mech} & {oc_det} & {oc_auc} & {oc_z} & "
            f"{oc_lead} & {mk_auc} & {mk_lead} & {adv_h} & {flip} \\\\"
        )

    header = r"""\begin{table}[t]
\centering
\small
\caption{Signal taxonomy: when on-chain AMM signals detect stablecoin stress.
Rows are grouped by shock origin (endogenous pool imbalance vs exogenous market shock).
``Lead'' is hours before the labelled panic onset; positive = fires before onset.
``Advantage'' is how many hours earlier the on-chain AMM signal fires relative to the CEX basis signal.}
\label{tab:taxonomy}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{4}{c}{On-chain AMM signal} & \multicolumn{2}{c}{CEX market signal} & & \\
\cmidrule(lr){3-6}\cmidrule(lr){7-8}
Event & Mechanism & Detected & AUROC & FR $z$ & Lead & AUROC & Lead & Adv.\ (h) & Arb.\ flip \\
\midrule
\multicolumn{10}{l}{\textit{Endogenous shock (pool-level origin)}} \\
"""
    mid = r"""\midrule
\multicolumn{10}{l}{\textit{Exogenous shock (external market origin)}} \\
"""
    footer = r"""\bottomrule
\end{tabular}}
\end{table}
"""

    tex_lines = [header]
    for r in endo:
        tex_lines.append(_fmt_row(r))
    tex_lines.append(mid)
    for r in exo:
        tex_lines.append(_fmt_row(r))
    tex_lines.append(footer)
    latex_str = "\n".join(tex_lines)

    tex_path = fdir / "fig_signal_taxonomy.tex"
    tex_path.write_text(latex_str, encoding="utf-8")

    # Log summary
    logger.info("=== Signal taxonomy ===")
    logger.info("%-22s  %-12s  %-10s  %-10s  %s",
                "event", "shock_type", "on-chain", "market", "advantage")
    for r in taxonomy_rows:
        logger.info("%-22s  %-12s  AUROC=%-6s  AUROC=%-6s  %sh",
                    r["event_id"], r["shock_type"],
                    r["onchain_hmm_auroc"] or "---",
                    r["market_auroc"] or "---",
                    r["onchain_earlier_by_h"] or "---")

    n_endo_det = sum(1 for r in endo if r["onchain_hmm_detected"])
    n_exo_det  = sum(1 for r in exo  if r["onchain_hmm_detected"])
    logger.info("Endogenous: %d/%d on-chain detected  |  Exogenous: %d/%d on-chain detected",
                n_endo_det, len(endo), n_exo_det, len(exo))
    logger.info("Wrote %s and %s", csv_path, tex_path)


if __name__ == "__main__":
    main()
