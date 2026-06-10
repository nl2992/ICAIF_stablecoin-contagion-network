"""Composite multi-panel figures to fit the 8-page ICAIF limit.
  fig_findings.pdf : 3-panel row (regime switch | arbitrage flip | price discovery)
  fig_ml.pdf       : 2-panel row (cross-event transfer matrix | HMM posterior)
Reads committed result tables + the usdt_curve gold panel.
"""
from __future__ import annotations
import warnings, numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger
warnings.filterwarnings("ignore"); logger = get_logger(__name__)

FIG = results_root() / "paper" / "figures"; TAB = results_root() / "tables"
_NAVY_INK = "#0A1F44"  # Columbia navy ink for axes / text
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "text.color": _NAVY_INK, "axes.labelcolor": _NAVY_INK,
                     "axes.titlecolor": _NAVY_INK, "axes.edgecolor": _NAVY_INK,
                     "xtick.color": _NAVY_INK, "ytick.color": _NAVY_INK})
# Refined Columbia palette: calm/baseline = Columbia navy; panic/pos/neg stay semantic.
C_CALM, C_PAN, C_POS, C_NEG = "#1D4F91", "#E0533D", "#27AE60", "#C0392B"
S = {"usdt_curve_2023":"USDT/\nCurve","terra_luna_2022":"Terra","ftx_2022":"FTX",
     "busd_2023":"BUSD","usdc_svb_2023":"USDC/\nSVB"}

def _save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f"{name}.pdf"); fig.savefig(FIG/f"{name}.png"); plt.close(fig)
    logger.info("wrote %s",name)

def findings():
    rg = pl.read_csv(TAB/"table_regime_contagion.csv")
    ar = pl.read_csv(TAB/"table_arbitrage_regime.csv")
    pd_ = pl.read_csv(TAB/"table_price_discovery.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.25))
    # panel A: regime contagion
    ev = rg["event_id"].to_list(); x=np.arange(len(ev)); w=.38
    ax[0].bar(x-w/2, rg["rho_pre"], w, label="calm", color=C_CALM)
    ax[0].bar(x+w/2, rg["rho_panic"], w, label="panic", color=C_PAN)
    for i,r in enumerate(rg.iter_rows(named=True)):
        if r["contagion_regime_shift"]: ax[0].text(x[i]+w/2,(r["rho_panic"] or 0)+.03,"*",ha="center",fontsize=12)
    ax[0].axhline(0,color="#888",lw=.6); ax[0].set_xticks(x); ax[0].set_xticklabels([S[e] for e in ev],fontsize=6.5)
    ax[0].set_ylabel(r"$\hat{\rho}$ cross-pool"); ax[0].set_title("(a) Regime contagion"); ax[0].legend(frameon=False,fontsize=6.5)
    # panel B: arbitrage flip
    ev=ar["event_id"].to_list(); x=np.arange(len(ev))
    rp=[v if v is not None else np.nan for v in ar["r_panic"]]
    ax[1].bar(x-w/2, ar["r_calm"], w, label="calm", color=C_CALM)
    ax[1].bar(x+w/2, rp, w, color=[C_POS if (v or 0)>0 else C_NEG for v in rp])
    ax[1].axhline(0,color="#333",lw=.7); ax[1].set_xticks(x); ax[1].set_xticklabels([S[e] for e in ev],fontsize=6.5)
    ax[1].set_ylabel("flow$\\leftrightarrow$price"); ax[1].set_title("(b) Arbitrage flip")
    # panel C: price discovery
    ev=pd_["event_id"].to_list(); x=np.arange(len(ev))
    ax[2].bar(x-w/2, pd_["onchain_leads_mean_r"], w, label="on-chain leads", color=C_POS)
    ax[2].bar(x+w/2, pd_["cex_leads_mean_r"], w, label="CEX leads", color=C_PAN)
    ax[2].set_xticks(x); ax[2].set_xticklabels([S[e] for e in ev],fontsize=6.5)
    ax[2].set_ylabel("lead corr."); ax[2].set_title("(c) Price discovery"); ax[2].legend(frameon=False,fontsize=6)
    fig.tight_layout(w_pad=1.1); _save(fig,"fig_findings")

def ml():
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35),
                           gridspec_kw={"width_ratios": [1.05, 1.0, 1.25]})

    # ---- panel (a): informational value -- Tier-B market vs +Tier-A on-chain ----
    iv = pl.read_csv(TAB/"table_informational_value.csv")
    ev_iv = iv["event_id"].to_list(); x = np.arange(len(ev_iv)); w = 0.38
    mk = iv["auroc_market"].to_list(); bo = iv["auroc_both"].to_list()
    ax[0].bar(x-w/2, mk, w, label="market only (B)", color=C_CALM)
    ax[0].bar(x+w/2, bo, w, label="+ on-chain (A)", color=C_POS)
    for i, r in enumerate(iv.iter_rows(named=True)):
        if r["lift_both_minus_market"] >= 0.05:
            ax[0].annotate(f"+{r['lift_both_minus_market']:.2f}", (x[i]+w/2, bo[i]+0.01),
                           ha="center", fontsize=6, color="#1d6b35")
    ax[0].axhline(0.5, ls="--", color="#999", lw=0.6)
    ax[0].set_xticks(x); ax[0].set_xticklabels([S[e] for e in ev_iv], fontsize=6.2)
    ax[0].set_ylim(0.45, 1.04); ax[0].set_ylabel("detection AUROC")
    ax[0].set_title("(a) On-chain adds info"); ax[0].legend(frameon=False, fontsize=5.8, loc="lower left")

    # ---- panel (b): cross-event transfer matrix (concept shift) ----
    df = pl.read_csv(TAB/"table_transfer_matrix.csv")
    ev=[c for c in df.columns if c!="train_event"]
    M=np.array([[df.filter(pl.col("train_event")==tr)[te][0] for te in ev] for tr in ev],float)
    im=ax[1].imshow(M,cmap="RdYlGn",vmin=0,vmax=1)
    ax[1].set_xticks(range(len(ev))); ax[1].set_yticks(range(len(ev)))
    lbl=[S[e].replace("\n","") for e in ev]
    ax[1].set_xticklabels(lbl,rotation=45,ha="right",fontsize=5.6); ax[1].set_yticklabels(lbl,fontsize=5.6)
    for i in range(len(ev)):
        for j in range(len(ev)): ax[1].text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=5.4)
    ax[1].set_xlabel("test",fontsize=7); ax[1].set_ylabel("train",fontsize=7)
    ax[1].set_title("(b) Transfer fails")

    # ---- panel (c): unsupervised HMM posterior ----
    from hmmlearn.hmm import GaussianHMM
    d=(pl.read_parquet(gold_root()/"dataset_contagion_features_usdt_curve_2023.parquet")
       .filter(pl.col("node_id")=="curve_3pool").with_columns((pl.col("event_time_seconds")//3600).alias("h"))
       .group_by("h").agg(pl.col("usdc_net_sold_1h").sum().alias("f"),pl.col("implied_pool_price").mean().alias("px"),
                          pl.col("reserve_imbalance").mean().alias("imb"),pl.col("event_phase").first().alias("ph")).sort("h"))
    h=d["h"].to_numpy()
    X=np.column_stack([np.abs(np.nan_to_num(d["f"].to_numpy())),np.abs(np.nan_to_num(d["px"].to_numpy(),nan=1.)-1.),
                       np.abs(np.nan_to_num(d["imb"].to_numpy()))]); X=(X-X.mean(0))/(X.std(0)+1e-9)
    y=(d["ph"].to_numpy()=="panic").astype(int)
    m=GaussianHMM(n_components=3,covariance_type="diag",n_iter=300,random_state=0).fit(X)
    post=m.predict_proba(X)[:,int(np.argmax(m.means_[:,1]))]
    inp=False
    for i in range(len(h)):
        if y[i] and not inp: s0=h[i]; inp=True
        if inp and (i==len(h)-1 or not y[i]): ax[2].axvspan(s0,h[i],color=C_PAN,alpha=.12); inp=False
    ax[2].plot(h,post,color="#222",lw=1.2); ax[2].axhline(.5,ls="--",color="#999",lw=.6)
    ax[2].set_xlabel("hours rel. onset",fontsize=7); ax[2].set_ylabel("P(stress)",fontsize=7); ax[2].set_ylim(-.03,1.03)
    ax[2].set_title("(c) HMM, no labels")
    fig.tight_layout(w_pad=0.8); _save(fig,"fig_ml")

def main():
    for fn in (findings, ml):
        try: fn()
        except Exception as e: logger.error("%s: %s",fn.__name__,e,exc_info=True)

if __name__=="__main__": main()
