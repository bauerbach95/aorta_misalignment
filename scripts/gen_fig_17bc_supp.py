#!/usr/bin/env python3
"""Generate Supplementary Figure 17 panels b and c (downsampled version).

Replicates the analysis from assess_wt_vs_ko_sex_similarity.ipynb using
downsampled data where all 4 conditions (male/female × WT/KO) are matched
for cell counts and UMIs.

Panel b: For each confident WT cycler, compute a cross-sex log-likelihood:
         sample from the female posterior, evaluate under the male posterior
         (as a normalized distance). X = WT metric, Y = KO metric.
         Points above y=x → female is more similar to male in KO.
Panel c: Fraction of genes where female is more similar to male in KO
         than in WT, at various BF thresholds for defining WT cyclers.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from circadian_utils import apply_style, FIGURES_DIR

apply_style()

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_ROOT = "/Users/mingyaolab/Dropbox/aorta_circadian_data"
DOWNSAMPLED_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05/male_cell_type_rhythmicity_same_lib_size_and_cell_count",
)

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_17bc_source_data")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

CLUSTER = "cluster_0"
NUM_SAMPLES = 300
FRAC_CELL_DETECTED_THRESH = 0.2  # same as original notebook


def load_params(cluster, condition):
    base = os.path.join(DOWNSAMPLED_DIR, cluster, condition)
    alpha = pd.read_table(os.path.join(base, "gene_log_alpha.tsv"), index_col="gene")
    beta = pd.read_table(os.path.join(base, "gene_log_beta.tsv"), index_col="gene")
    minmax = pd.read_table(os.path.join(base, "log_min_max.tsv"), index_col="gene")
    stats = pd.read_table(os.path.join(base, "de_novo_metrics.tsv"), index_col="gene")
    return alpha, beta, minmax, stats


def sample_waveforms(alpha_df, beta_df, minmax_df, gene_list, n=NUM_SAMPLES):
    """Sample posterior waveforms. Returns [n_samples, n_timepoints, n_genes]."""
    genes = [g for g in gene_list if g in alpha_df.index]
    alpha = np.exp(alpha_df.loc[genes].values)   # [n_genes, 4]
    beta_v = np.exp(beta_df.loc[genes].values)
    log_min = minmax_df.loc[genes, "log_min"].values
    log_max = minmax_df.loc[genes, "log_max"].values

    dist = torch.distributions.Beta(
        torch.tensor(alpha.T, dtype=torch.float32),
        torch.tensor(beta_v.T, dtype=torch.float32),
    )
    raw = dist.sample((n,)).numpy()  # [n_samples, 4, n_genes]
    raw = raw * (log_max[None, None, :] - log_min[None, None, :]) + log_min[None, None, :]
    return raw  # [n_samples, 4, n_genes]


# ── Load parameters ──────────────────────────────────────────────────────────

print("Loading downsampled parameters...")
a_mwt, b_mwt, mm_mwt, s_mwt = load_params(CLUSTER, "male aligned bmal1-control")
a_fwt, b_fwt, mm_fwt, s_fwt = load_params(CLUSTER, "female aligned bmal1-control")
a_mko, b_mko, mm_mko, s_mko = load_params(CLUSTER, "male aligned bmal1-ko")
a_fko, b_fko, mm_fko, s_fko = load_params(CLUSTER, "female aligned bmal1-ko")

# ── Filter genes ─────────────────────────────────────────────────────────────

BF_COL = "waveform_over_circadian_component_subtracted_log10_bf"

# Filter by detection (must be detected in at least one sex × genotype)
genes_to_filter_m = set(s_mwt[s_mwt["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
genes_to_filter_m.update(s_mko[s_mko["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
genes_to_filter_f = set(s_fwt[s_fwt["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
genes_to_filter_f.update(s_fko[s_fko["frac_cell_detected"] < FRAC_CELL_DETECTED_THRESH].index)
genes_to_filter = genes_to_filter_m.intersection(genes_to_filter_f)

# Hemoglobin genes
hb_genes = set(g for g in s_mwt.index if g[:2] == "Hb" and "-" in g)
genes_to_filter.update(hb_genes)

common_genes = sorted(
    set(a_mwt.index) & set(a_fwt.index) & set(a_mko.index) & set(a_fko.index)
    - genes_to_filter
)
print(f"Common genes after filtering: {len(common_genes)}")

# ── Run analysis at multiple BF thresholds ────────────────────────────────────

bf_thresholds = [1, 2, 5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300]
frac_more_similar = []
scatter_data = None

for bf_thresh in bf_thresholds:
    male_wt_cyclers = set(s_mwt[s_mwt[BF_COL] >= bf_thresh].index)
    female_wt_cyclers = set(s_fwt[s_fwt[BF_COL] >= bf_thresh].index)
    wt_cyclers = male_wt_cyclers.union(female_wt_cyclers)
    cycler_genes = sorted(set(common_genes) & wt_cyclers)

    if len(cycler_genes) == 0:
        frac_more_similar.append(np.nan)
        continue

    # Sample waveforms for these genes
    slp_mwt = sample_waveforms(a_mwt, b_mwt, mm_mwt, cycler_genes)
    slp_fwt = sample_waveforms(a_fwt, b_fwt, mm_fwt, cycler_genes)
    slp_mko = sample_waveforms(a_mko, b_mko, mm_mko, cycler_genes)
    slp_fko = sample_waveforms(a_fko, b_fko, mm_fko, cycler_genes)

    # Male KO effect CI (used as normalization, same as original)
    male_ko_effect = slp_mko - slp_mwt  # [n_samples, 4, n_genes]
    ci_low = np.percentile(male_ko_effect, 2.5, axis=0)  # [4, n_genes]
    ci_high = np.percentile(male_ko_effect, 97.5, axis=0)
    ci_width = ci_high - ci_low
    ci_width = np.maximum(ci_width, 1e-10)

    # WT: female-male difference, normalized by CI width
    wt_dif = slp_fwt - slp_mwt  # [n_samples, 4, n_genes]
    nll_wt = -1.0 * np.sum(np.sum(np.abs(wt_dif) / ci_width[None, :, :], axis=0), axis=0)

    # KO: female-male difference, normalized by CI width
    ko_dif = slp_fko - slp_mko
    nll_ko = -1.0 * np.sum(np.sum(np.abs(ko_dif) / ci_width[None, :, :], axis=0), axis=0)

    frac = np.mean(nll_ko > nll_wt)
    frac_more_similar.append(frac)
    print(f"  BF>={bf_thresh}: {len(cycler_genes)} cyclers, frac more similar in KO: {frac:.4f}")

    # Save scatter data for the first run with BF >= 100 (matching original)
    if bf_thresh == 100 and scatter_data is None:
        scatter_data = {
            "genes": cycler_genes,
            "nll_wt": nll_wt,
            "nll_ko": nll_ko,
        }

# ── Panel b: Scatter plot ────────────────────────────────────────────────────

if scatter_data is None:
    print("WARNING: No scatter data at BF>=100, using BF>=2 instead")
    # Fallback: recompute at BF>=2
    bf_thresh = 2
    male_wt_cyclers = set(s_mwt[s_mwt[BF_COL] >= bf_thresh].index)
    female_wt_cyclers = set(s_fwt[s_fwt[BF_COL] >= bf_thresh].index)
    wt_cyclers = male_wt_cyclers.union(female_wt_cyclers)
    cycler_genes = sorted(set(common_genes) & wt_cyclers)
    slp_mwt = sample_waveforms(a_mwt, b_mwt, mm_mwt, cycler_genes)
    slp_fwt = sample_waveforms(a_fwt, b_fwt, mm_fwt, cycler_genes)
    slp_mko = sample_waveforms(a_mko, b_mko, mm_mko, cycler_genes)
    slp_fko = sample_waveforms(a_fko, b_fko, mm_fko, cycler_genes)
    male_ko_effect = slp_mko - slp_mwt
    ci_low = np.percentile(male_ko_effect, 2.5, axis=0)
    ci_high = np.percentile(male_ko_effect, 97.5, axis=0)
    ci_width = np.maximum(ci_high - ci_low, 1e-10)
    wt_dif = slp_fwt - slp_mwt
    nll_wt = -1.0 * np.sum(np.sum(np.abs(wt_dif) / ci_width[None, :, :], axis=0), axis=0)
    ko_dif = slp_fko - slp_mko
    nll_ko = -1.0 * np.sum(np.sum(np.abs(ko_dif) / ci_width[None, :, :], axis=0), axis=0)
    scatter_data = {"genes": cycler_genes, "nll_wt": nll_wt, "nll_ko": nll_ko}

nll_wt = scatter_data["nll_wt"]
nll_ko = scatter_data["nll_ko"]
genes = scatter_data["genes"]

# Linear fit
lr = LinearRegression(fit_intercept=True)
lr.fit(nll_wt.reshape(-1, 1), nll_ko)
y_hat = lr.predict(nll_wt.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(2.2, 1.8), dpi=300)
ax.scatter(nll_wt, nll_ko, s=1, alpha=0.6, color="grey", rasterized=True)
sort_idx = np.argsort(nll_wt)
ax.plot(nll_wt[sort_idx], y_hat[sort_idx], c="r", linestyle="--", alpha=0.75,
        linewidth=0.8, label="best fit")
ax.plot(nll_wt[sort_idx], nll_wt[sort_idx], c="k", alpha=0.75,
        linewidth=0.8, label="y=x")
ax.set_xlabel("WT: LL of female samples\nin male distribution")
ax.set_ylabel("KO: LL of female samples\nin male distribution")
ax.legend(fontsize=4, loc="upper left")
sns.despine(ax=ax)
plt.tight_layout()

outpath_b = os.path.join(FIGURES_DIR, "fig_17b_supp.pdf")
fig.savefig(outpath_b, bbox_inches="tight", dpi=300)
print(f"\nPanel b saved to {outpath_b}")

frac_above = np.mean(nll_ko > nll_wt)
print(f"  Genes: {len(genes)}, fraction more similar in KO: {frac_above:.4f}")

# ── Panel c: Fraction more similar at various thresholds ──────────────────────

fig, ax = plt.subplots(figsize=(2.2, 1.8), dpi=300)
ax.scatter(bf_thresholds, frac_more_similar, s=3, color="#3C5488")
ax.set_xlabel("Log10 bayes factor threshold\nto call WT cyclers")
ax.set_ylabel("Fraction of genes more\nlikely in KO than WT")
ax.set_xlim(-10, 310)
ax.set_ylim(0.5, max(frac_more_similar) + 0.05)
sns.despine(ax=ax)
plt.tight_layout()

outpath_c = os.path.join(FIGURES_DIR, "fig_17c_supp.pdf")
fig.savefig(outpath_c, bbox_inches="tight", dpi=300)
print(f"Panel c saved to {outpath_c}")

# ── Source data ──────────────────────────────────────────────────────────────

source_b = pd.DataFrame({
    "gene": genes,
    "wt_cross_sex_ll": nll_wt,
    "ko_cross_sex_ll": nll_ko,
})
source_b.to_csv(os.path.join(SOURCE_DATA_DIR, "panel_b_cross_sex_ll.csv"), index=False)

source_c = pd.DataFrame({
    "bf_threshold": bf_thresholds,
    "frac_more_similar_in_ko": frac_more_similar,
})
source_c.to_csv(os.path.join(SOURCE_DATA_DIR, "panel_c_frac_more_similar.csv"), index=False)

print(f"Source data written to {SOURCE_DATA_DIR}")
