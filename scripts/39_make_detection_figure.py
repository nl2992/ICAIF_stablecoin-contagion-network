"""Figure for the online causal detection benchmark.
  (a) causal AUROC on-chain vs market per event (data-source comparison)
  (b) Terra/LUNA filtered stress posterior: on-chain rises at onset, the market
      detector stays flat (the CEX USDT basis never moves during a UST de-peg).
Writes results/paper/figures/fig_detection.{pdf,png}
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm
from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger
warnings.filterwarnings("ignore"); logger = get_logger(__name__)

import importlib.util, sys
spec = importlib.util.spec_from_file_location("d38", "scripts/38_online_detection_benchmark.py")
d38 = importlib.util.module_from_spec(spec); sys.modules["d38"] = d38; spec.loader.exec_module(d38)

FIG = results_root() / "paper" / "figures"; TAB = results_root() / "tables"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_OC, C_MK = "#27AE60", "#5B8DEF"
S = {"usdt_curve_2023": "USDT/\nCurve", "terra_luna_2022": "Terra", "ftx_2022": "FTX",
     "busd_2023": "BUSD", "usdc_svb_2023": "USDC/\nSVB"}


def main():
    t = pl.read_csv(TAB / "table_online_detection.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # panel (a): causal AUROC bars
    ev = t["event"].to_list(); x = np.arange(len(ev)); w = 0.38
    ax[0].bar(x - w/2, t["auroc_onchain_causal"], w, label="on-chain (A)", color=C_OC)
    ax[0].bar(x + w/2, t["auroc_market_causal"], w, label="market (B)", color=C_MK)
    ax[0].axhline(0.5, ls="--", color="#999", lw=0.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels([S[e] for e in ev], fontsize=6.5)
    ax[0].set_ylim(0.35, 1.02); ax[0].set_ylabel("causal AUROC")
    ax[0].set_title("(a) Online detection by data source")
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower center", ncol=2)

    # panel (b): Terra underlying z-scored signals -- why on-chain detects and
    # market cannot.  Pool price-deviation rises through the panic window; the
    # USDT/Binance basis does not move (USDT stayed pegged in a UST de-peg).
    onchain, market, y = d38._grid("terra_luna_2022")
    onset = int(np.argmax(y == 1))
    def z(v):
        v = v.astype(float); return (v - v.mean()) / (v.std() + 1e-9)
    sig_oc = z(onchain[:, 0])   # pool |price-dev|
    sig_mk = z(market[:, 0])    # CEX |basis|
    # light smoothing for readability
    def sm(a, k=9):
        return np.convolve(a, np.ones(k)/k, mode="same")
    h = np.arange(len(y)) - onset
    inp = False
    for i in range(len(y)):
        if y[i] and not inp: s0 = h[i]; inp = True
        if inp and (i == len(y)-1 or not y[i]): ax[1].axvspan(s0, h[i], color="#E0533D", alpha=.12, label="panic" if i < 3 else None); inp = False
    ax[1].plot(h, sm(sig_oc), color=C_OC, lw=1.5, label="on-chain pool dev. (A)")
    ax[1].plot(h, sm(sig_mk), color=C_MK, lw=1.2, label="CEX basis (B)")
    ax[1].axvline(0, color="#E0533D", ls="--", lw=0.8)
    ax[1].set_xlabel("hours rel. onset"); ax[1].set_ylabel("standardized signal ($z$)")
    ax[1].set_title("(b) Terra: pool dev. spikes, CEX basis does not")
    ax[1].legend(frameon=False, fontsize=6.2, loc="upper right")

    fig.tight_layout(w_pad=1.0)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_detection.pdf"); fig.savefig(FIG / "fig_detection.png")
    plt.close(fig); logger.info("wrote fig_detection")


if __name__ == "__main__":
    main()
