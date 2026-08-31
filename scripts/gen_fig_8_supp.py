"""
Generate Supplementary Figure 8: Female misalignment cycling controlled for cell
counts and library sizes.

Female counterpart to Supplementary Figure 5 (male).

Subplots:
  a) 2x2 grid: cycler counts vs log10 BF threshold for 4 major cell types (female)
  b) 1x3 grid: cycler counts vs log10 BF threshold for 3 SMC subtypes (female)

Output:
  figures/fig_8_supp.pdf
  figures/fig_8_supp_source_data/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circadian_utils import (
    apply_style, DATA_ROOT, CELL_TYPE_COLORS, FIGURES_DIR,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_8_supp_source_data")

CLUSTERING_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05",
)

MAJOR_DS_DIR = os.path.join(
    CLUSTERING_DIR, "male_vs_female_aligned_cell_type_rhythmicity",
)

SMC_SUBTYPE_DS_DIR = os.path.join(
    CLUSTERING_DIR, "subclustering/res_0.2/"
    "male_vs_female_aligned_SMC_subtype_rhythmicity",
)

MAJOR_CLUSTERS = {
    "cluster_0": "SMC",
    "cluster_1": "Fibroblast",
    "cluster_2": "EC",
    "cluster_3": "Macrophage",
}

SMC_SUBTYPE_CLUSTERS = {
    "cluster_0": "SMC0",
    "cluster_1": "SMC1",
    "cluster_2": "SMC2",
}

BF_COL = "waveform_over_circadian_component_subtracted_log10_bf"
COND_AL = "female aligned bmal1-control"
COND_MIS = "female misaligned bmal1-control"

BF_MIN = 0.5
BF_MAX = 10.0
NUM_POINTS = 1000

LINE_STYLES = {
    "Aligned": {"color": "#4878CF", "linewidth": 1.0, "linestyle": "-"},
    "Misaligned": {"color": "#D65F5F", "linewidth": 1.0, "linestyle": "-"},
    "Unique aligned": {"color": "#4878CF", "linewidth": 0.8, "linestyle": "--"},
    "Unique misaligned": {"color": "#D65F5F", "linewidth": 0.8, "linestyle": "--"},
    "Shared": {"color": "#6ACC65", "linewidth": 1.0, "linestyle": "-"},
}


def compute_cycler_counts(bf_al, bf_mis, bf_grid):
    rows = []
    for thresh in bf_grid:
        mask_al = bf_al >= thresh
        mask_mis = bf_mis >= thresh
        rows.append({
            "Aligned": int(mask_al.sum()),
            "Misaligned": int(mask_mis.sum()),
            "Unique aligned": int((mask_al & ~mask_mis).sum()),
            "Unique misaligned": int((~mask_al & mask_mis).sum()),
            "Shared": int((mask_al & mask_mis).sum()),
        })
    return pd.DataFrame(rows, index=bf_grid)


def load_bf_pair(base_dir, cluster):
    df_al = pd.read_csv(
        os.path.join(base_dir, cluster, COND_AL, "de_novo_metrics.tsv"),
        sep="\t", index_col="gene",
    )
    df_mis = pd.read_csv(
        os.path.join(base_dir, cluster, COND_MIS, "de_novo_metrics.tsv"),
        sep="\t", index_col="gene",
    )
    common = sorted(set(df_al.index) & set(df_mis.index))
    return df_al.loc[common, BF_COL], df_mis.loc[common, BF_COL]


def plot_cycler_panel(ax, counts_df, title, title_color="0.15", show_legend=False):
    for label, style in LINE_STYLES.items():
        ax.plot(counts_df.index, counts_df[label], label=label, **style)
    ax.set_xlim(BF_MIN, BF_MAX)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("Log$_{10}$ circadian BF threshold", fontsize=6)
    ax.set_ylabel("Number of cyclers", fontsize=6)
    ax.set_title(title, fontsize=7, fontweight="semibold", color=title_color)
    ax.tick_params(labelsize=5.5)
    sns.despine(ax=ax)
    if show_legend:
        ax.legend(fontsize=4.5, frameon=False, loc="upper right")


# ── Step 1: Compute cycler counts ──────────────────────────────────────────

print("Step 1: Computing cycler counts across BF thresholds (female)...")

bf_grid = np.linspace(BF_MIN, BF_MAX, NUM_POINTS)

major_counts = {}
for cluster, cell_type in MAJOR_CLUSTERS.items():
    bf_al, bf_mis = load_bf_pair(MAJOR_DS_DIR, cluster)
    major_counts[cell_type] = compute_cycler_counts(bf_al, bf_mis, bf_grid)
    n_al = int((bf_al >= 2).sum())
    n_mis = int((bf_mis >= 2).sum())
    print(f"  {cell_type}: {len(bf_al)} common genes, {n_al} aligned / {n_mis} misaligned cyclers (BF>=2)")

smc_counts = {}
for cluster, subtype in SMC_SUBTYPE_CLUSTERS.items():
    bf_al, bf_mis = load_bf_pair(SMC_SUBTYPE_DS_DIR, cluster)
    smc_counts[subtype] = compute_cycler_counts(bf_al, bf_mis, bf_grid)
    n_al = int((bf_al >= 2).sum())
    n_mis = int((bf_mis >= 2).sum())
    print(f"  {subtype}: {len(bf_al)} common genes, {n_al} aligned / {n_mis} misaligned cyclers (BF>=2)")

# ── Step 2: Export source data ─────────────────────────────────────────────

print("\nStep 2: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

for name, counts_df in {**major_counts, **smc_counts}.items():
    out = counts_df.copy()
    out.index.name = "bf_threshold"
    out.to_csv(os.path.join(SOURCE_DATA_DIR, f"cycler_counts_{name}.csv"))

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 3: Generate figure ────────────────────────────────────────────────

print("\nStep 3: Generating figure...")

fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
fig.patch.set_facecolor("white")

gs_outer = GridSpec(2, 1, figure=fig, hspace=0.45,
                    height_ratios=[2, 1],
                    left=0.08, right=0.96, top=0.95, bottom=0.08)

# Panel a: 2x2 major cell types
gs_a = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[0],
                               hspace=0.50, wspace=0.35)
major_order = ["SMC", "Fibroblast", "EC", "Macrophage"]
for i, cell_type in enumerate(major_order):
    r, c = divmod(i, 2)
    ax = fig.add_subplot(gs_a[r, c])
    color = CELL_TYPE_COLORS[cell_type]
    plot_cycler_panel(ax, major_counts[cell_type], cell_type,
                      title_color=color, show_legend=(i == 0))
    if i == 0:
        ax.text(-0.20, 1.18, "a", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Panel b: 1x3 SMC subtypes
gs_b = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1], wspace=0.35)
smc_order = ["SMC0", "SMC1", "SMC2"]
smc_colors = plt.cm.Reds(np.linspace(0.3, 0.85, 3))

for j, subtype in enumerate(smc_order):
    ax = fig.add_subplot(gs_b[0, j])
    plot_cycler_panel(ax, smc_counts[subtype], subtype,
                      title_color=smc_colors[j])
    if j == 0:
        ax.text(-0.20, 1.18, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_8_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
