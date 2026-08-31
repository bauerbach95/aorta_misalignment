"""
Generate Supplementary Figure 19: Bmal1 KO cycling controlled for sequencing depth,
plus sex specificity mediation by the clock.

Panel a: Male WT vs iKO cycler counts across BF thresholds (SMC, Fibroblast)
Panel b: Female WT vs iKO cycler counts across BF thresholds (SMC, Fibroblast)
Panel c: Cross-sex log-likelihood scatter — all detected genes, cyclers highlighted

Uses downsampled data (matched cell counts and library sizes across all 4 conditions).

Output:
  figures/fig_19_supp.pdf
  figures/fig_19_source_data/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from circadian_utils import (
    DATA_ROOT, FIGURES_DIR,
    FRAC_CELL_DETECTED_MIN, NUM_CELL_DETECTED_MIN,
    FRAC_CIRC_LARGEST_COMP_MIN, WAVEFORM_BF_MIN,
    CELL_TYPE_COLORS, apply_style,
)

apply_style()

DOWNSAMPLED_KO_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05/male_cell_type_rhythmicity_same_lib_size_and_cell_count",
)

CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}

CONDITIONS = {
    "male": {
        "wt": "male aligned bmal1-control",
        "ko": "male aligned bmal1-ko",
    },
    "female": {
        "wt": "female aligned bmal1-control",
        "ko": "female aligned bmal1-ko",
    },
}

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_19_source_data")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

# ── Panel c parameters ──────────────────────────────────────────────────

PANEL_C_CLUSTER = "cluster_0"
NUM_SAMPLES = 300
FRAC_CELL_DETECTED_THRESH = 0.2
BF_CYCLER_THRESH = 10
BF_COL = "waveform_over_circadian_component_subtracted_log10_bf"


def load_metrics(cluster, condition):
    path = os.path.join(DOWNSAMPLED_KO_DIR, cluster, condition, "de_novo_metrics.tsv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Downsampled data missing: {path}\n"
            f"Run create_downsampled_fibroblast_ko_data.py first."
        )
    return pd.read_table(path, sep="\t", index_col="gene")


def filter_cyclers(metric_df, bf_threshold=WAVEFORM_BF_MIN):
    mask = (
        (metric_df["frac_cell_detected"] > FRAC_CELL_DETECTED_MIN)
        & (metric_df["num_cell_detected"] > NUM_CELL_DETECTED_MIN)
        & (metric_df["frac_circadian_samples_largest_component"] >= FRAC_CIRC_LARGEST_COMP_MIN)
        & (metric_df["waveform_over_circadian_component_subtracted_log10_bf"] >= bf_threshold)
    )
    return set(metric_df.index[mask])


def load_params(cluster, condition):
    base = os.path.join(DOWNSAMPLED_KO_DIR, cluster, condition)
    alpha = pd.read_table(os.path.join(base, "gene_log_alpha.tsv"), index_col="gene")
    beta = pd.read_table(os.path.join(base, "gene_log_beta.tsv"), index_col="gene")
    minmax = pd.read_table(os.path.join(base, "log_min_max.tsv"), index_col="gene")
    stats = pd.read_table(os.path.join(base, "de_novo_metrics.tsv"), index_col="gene")
    return alpha, beta, minmax, stats


def sample_waveforms_flat(a_df, b_df, mm_df, genes, n=NUM_SAMPLES):
    """Sample posterior waveforms. Returns [n_samples, n_timepoints, n_genes]."""
    a = np.exp(a_df.loc[genes].values)
    b = np.exp(b_df.loc[genes].values)
    lmin = mm_df.loc[genes, "log_min"].values
    lmax = mm_df.loc[genes, "log_max"].values
    dist = torch.distributions.Beta(
        torch.tensor(a.T, dtype=torch.float32),
        torch.tensor(b.T, dtype=torch.float32),
    )
    raw = dist.sample((n,)).numpy()
    return raw * (lmax[None, None, :] - lmin[None, None, :]) + lmin[None, None, :]


# ══════════════════════════════════════════════════════════════════════════
# PANELS A & B: Cycler count sweeps
# ══════════════════════════════════════════════════════════════════════════

bf_thresholds = np.concatenate([np.arange(0.5, 3.0, 0.25), np.arange(3, 11, 1.0)])

print("Computing rhythmicity sweeps (WT vs iKO, downsampled)...")

sweep_data = {}
for sex, conds in CONDITIONS.items():
    sex_data = {}
    for cluster, cell_type in CLUSTERS.items():
        wt_df = load_metrics(cluster, conds["wt"])
        ko_df = load_metrics(cluster, conds["ko"])

        results = []
        for thresh in bf_thresholds:
            wt_cyc = filter_cyclers(wt_df, bf_threshold=thresh)
            ko_cyc = filter_cyclers(ko_df, bf_threshold=thresh)
            shared = wt_cyc & ko_cyc
            results.append({
                "bf_threshold": thresh,
                "wt": len(wt_cyc),
                "ko": len(ko_cyc),
                "shared": len(shared),
            })
        sex_data[cell_type] = pd.DataFrame(results)
        print(f"  {sex} {cell_type}: WT={results[0]['wt']}, KO={results[0]['ko']} at BF>={bf_thresholds[0]}")
    sweep_data[sex] = sex_data

for sex, sex_data in sweep_data.items():
    panel = "a" if sex == "male" else "b"
    for ct, df in sex_data.items():
        df.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_{panel}_sweep_{sex}_{ct}.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════
# PANEL C: Cross-sex log-likelihood (sex specificity mediation)
# ══════════════════════════════════════════════════════════════════════════

print("\nComputing cross-sex log-likelihood for panel c...")

a_mwt, b_mwt, mm_mwt, s_mwt = load_params(PANEL_C_CLUSTER, "male aligned bmal1-control")
a_fwt, b_fwt, mm_fwt, s_fwt = load_params(PANEL_C_CLUSTER, "female aligned bmal1-control")
a_mko, b_mko, mm_mko, s_mko = load_params(PANEL_C_CLUSTER, "male aligned bmal1-ko")
a_fko, b_fko, mm_fko, s_fko = load_params(PANEL_C_CLUSTER, "female aligned bmal1-ko")

# Filter genes by detection in both sexes
gf_m = set(s_mwt[s_mwt["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
gf_m |= set(s_mko[s_mko["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
gf_f = set(s_fwt[s_fwt["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
gf_f |= set(s_fko[s_fko["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
genes_to_filter = gf_m & gf_f
hb_genes = set(g for g in s_mwt.index if g[:2] == "Hb" and "-" in g)
genes_to_filter |= hb_genes

common_genes = sorted(
    set(a_mwt.index) & set(a_fwt.index) & set(a_mko.index) & set(a_fko.index) - genes_to_filter
)
print(f"  Common genes after filtering: {len(common_genes)}")

wt_cyc = set(s_mwt[s_mwt[BF_COL] >= BF_CYCLER_THRESH].index)
wt_cyc |= set(s_fwt[s_fwt[BF_COL] >= BF_CYCLER_THRESH].index)
is_cycler = np.array([g in wt_cyc for g in common_genes])

print("  Sampling waveforms...")
slp_mwt = sample_waveforms_flat(a_mwt, b_mwt, mm_mwt, common_genes)
slp_fwt = sample_waveforms_flat(a_fwt, b_fwt, mm_fwt, common_genes)
slp_mko = sample_waveforms_flat(a_mko, b_mko, mm_mko, common_genes)
slp_fko = sample_waveforms_flat(a_fko, b_fko, mm_fko, common_genes)

ko_eff = slp_mko - slp_mwt
ci_lo = np.percentile(ko_eff, 2.5, axis=0)
ci_hi = np.percentile(ko_eff, 97.5, axis=0)
ci_w = np.maximum(ci_hi - ci_lo, 1e-10)

nll_wt = -np.sum(np.sum(np.abs(slp_fwt - slp_mwt) / ci_w[None, :, :], axis=0), axis=0)
nll_ko = -np.sum(np.sum(np.abs(slp_fko - slp_mko) / ci_w[None, :, :], axis=0), axis=0)

n_cyc = is_cycler.sum()
n_non = (~is_cycler).sum()
print(f"  Cyclers: {n_cyc}, Non-cyclers: {n_non}")
print(f"  Cycler frac above y=x: {np.mean(nll_ko[is_cycler] > nll_wt[is_cycler]):.4f}")
print(f"  Non-cycler frac above y=x: {np.mean(nll_ko[~is_cycler] > nll_wt[~is_cycler]):.4f}")

source_c = pd.DataFrame({
    "gene": common_genes,
    "wt_cross_sex_ll": nll_wt,
    "ko_cross_sex_ll": nll_ko,
    "is_wt_cycler": is_cycler,
})
source_c.to_csv(os.path.join(SOURCE_DATA_DIR, "panel_c_cross_sex_ll.csv"), index=False)
print(f"Source data written to {SOURCE_DATA_DIR}/")

# ══════════════════════════════════════════════════════════════════════════
# ASSEMBLE FIGURE
# ══════════════════════════════════════════════════════════════════════════

print("\nGenerating figure...")

fig = plt.figure(figsize=(6.5, 7.5), dpi=300)
fig.patch.set_facecolor("white")

gs_main = GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                   left=0.10, right=0.72, top=0.95, bottom=0.06,
                   height_ratios=[1, 1, 1.3])

line_colors = {"wt": "#2C7BB6", "ko": "#D7191C", "shared": "#9B59B6"}
cell_types_ordered = ["SMC", "Fibroblast"]

# ── Panels a & b: cycler count sweeps ────────────────────────────────────

for row_idx, (sex, sex_data) in enumerate(sweep_data.items()):
    panel_label = "a" if sex == "male" else "b"
    for col_idx, ct in enumerate(cell_types_ordered):
        ax = fig.add_subplot(gs_main[row_idx, col_idx])
        df = sex_data[ct]

        ax.plot(df["bf_threshold"], df["wt"], color=line_colors["wt"],
                linewidth=1.2, label="Aligned WT")
        ax.plot(df["bf_threshold"], df["ko"], color=line_colors["ko"],
                linewidth=1.2, label="Aligned iKO")
        ax.plot(df["bf_threshold"], df["shared"], color=line_colors["shared"],
                linewidth=1.0, linestyle="--", label="Shared", alpha=0.7)

        ax.set_title(f"{ct} ({sex})", fontsize=7, fontweight="semibold",
                     color=CELL_TYPE_COLORS[ct])
        ax.set_xlabel(r"$\log_{10}$ circadian Bayes factor threshold", fontsize=6)
        if col_idx == 0:
            ax.set_ylabel("Number of cyclers", fontsize=6.5)
        ax.tick_params(axis="both", labelsize=5.5)
        ax.legend(fontsize=4.5, frameon=False, loc="upper right")
        sns.despine(ax=ax)

        if col_idx == 0:
            ax.text(-0.18, 1.12, panel_label, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")

# ── Panel c: cross-sex log-likelihood scatter ────────────────────────────

ax_c = fig.add_subplot(gs_main[2, 0])
ax_c.set_aspect("equal", adjustable="box")
ax_c.set_box_aspect(1)

ax_c.scatter(nll_wt[~is_cycler], nll_ko[~is_cycler], s=3, alpha=0.4,
             color="grey", rasterized=True, label=f"non-cycler ({n_non})")
ax_c.scatter(nll_wt[is_cycler], nll_ko[is_cycler], s=3, alpha=0.6,
             color="#E64B35", rasterized=True, label=f"WT cycler BF≥{BF_CYCLER_THRESH} ({n_cyc})")

lr_cyc = LinearRegression().fit(nll_wt[is_cycler].reshape(-1, 1), nll_ko[is_cycler])
lr_non = LinearRegression().fit(nll_wt[~is_cycler].reshape(-1, 1), nll_ko[~is_cycler])
si_cyc = np.argsort(nll_wt[is_cycler])
si_non = np.argsort(nll_wt[~is_cycler])
yh_cyc = lr_cyc.predict(nll_wt[is_cycler].reshape(-1, 1))
yh_non = lr_non.predict(nll_wt[~is_cycler].reshape(-1, 1))

ax_c.plot(nll_wt[is_cycler][si_cyc], yh_cyc[si_cyc], c="r", linestyle="--",
          alpha=0.75, linewidth=0.8, label="best fit (cyclers)")
ax_c.plot(nll_wt[~is_cycler][si_non], yh_non[si_non], c="grey", linestyle="--",
          alpha=0.75, linewidth=0.8, label="best fit (non-cyclers)")
xr = np.array([-600, -200])
ax_c.plot(xr, xr, c="k", alpha=0.75, linewidth=0.8, label="y=x")

ticks = [-600, -500, -400, -300, -200]
ax_c.set_xlim(-600, -200)
ax_c.set_ylim(-600, -200)
ax_c.set_xticks(ticks)
ax_c.set_yticks(ticks)
ax_c.set_xlabel("WT: LL of female samples\nin male distribution", fontsize=6)
ax_c.set_ylabel("KO: LL of female samples\nin male distribution", fontsize=6)
ax_c.tick_params(axis="both", labelsize=5.5)
ax_c.legend(fontsize=4, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
sns.despine(ax=ax_c)
ax_c.text(-0.18, 1.12, "c", transform=ax_c.transAxes,
          fontsize=11, fontweight="bold", va="top")

outpath = os.path.join(FIGURES_DIR, "fig_19_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
