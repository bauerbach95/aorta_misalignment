"""
Generate Supplementary Figure 1: Validation of circadian cycling calls.

Subplots:
  a) Scatter: SMC acrophase (single-cell) vs bulk aorta (Zhang et al.)
  b) Violin: Core clock gene waveforms across 4 major cell types
  c) Violin: Core clock gene waveforms across 5 SMC subtypes
  d) Histograms: Acrophase distribution per major cell type

Output:
  figures/fig_1_supp.pdf
  figures/fig_1_supp_source_data/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, CELL_TYPE_COLORS,
    FIGURES_DIR, DATA_ROOT, NONPARAM_REG_DIR,
    load_metrics, filter_cyclers, acrophase_rad_to_hours,
    load_waveform_params, sample_waveforms, ZT_LABELS,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_1_supp_source_data")
NUM_WAVEFORM_SAMPLES = 300
CLOCK_GENES = ["Arntl", "Nfil3", "Dbp", "Nr1d2"]
CONDITION = "male aligned bmal1-control"

BULK_DATA_DIR = os.path.join(DATA_ROOT, "bulk_data")
BULK_JTK_PATH = os.path.join(BULK_DATA_DIR, "JTKresult_Aor.MetaCycle_input.txt")
BULK_PROBE_MAP_PATH = os.path.join(BULK_DATA_DIR, "BHTC_probe-to-gene-mapping.txt")

SMC_SUBCLUSTER_REG_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05/subclustering/res_0.2/nonparametric_reg",
)
SMC_SUBTYPE_CLUSTERS = {
    f"cluster_{i}": f"SMC{i}" for i in range(5)
}

# ── Step 1: Load bulk JTK results ─────────────────────────────────────────

print("Step 1: Loading bulk aorta JTK results...")
jtk_df = pd.read_table(BULK_JTK_PATH, sep="\t")

probe_map_df = pd.read_table(BULK_PROBE_MAP_PATH, sep="\t")
probe_map = dict(zip(probe_map_df["probeset_IDs"], probe_map_df["geneSymbol"]))
jtk_df["gene_symbol"] = jtk_df["CycID"].map(probe_map).fillna("None")
jtk_df = jtk_df.drop_duplicates(subset=["gene_symbol"], keep=False)
jtk_df = jtk_df.set_index("gene_symbol")
jtk_df = jtk_df[jtk_df["BH.Q"] <= 0.05]
jtk_df["ADJ_LAG"] = (jtk_df["LAG"] - jtk_df.loc["Arntl", "LAG"]) % 24
print(f"  {len(jtk_df)} significant bulk cyclers (BH.Q <= 0.05)")

# ── Step 2: Load SMC single-cell cyclers ──────────────────────────────────

print("\nStep 2: Loading SMC single-cell cyclers...")
smc_metrics = load_metrics("cluster_0", CONDITION)
smc_cyclers = filter_cyclers(smc_metrics)
smc_metrics_cyc = smc_metrics.loc[sorted(smc_cyclers)]
smc_metrics_cyc["acrophase_hours"] = acrophase_rad_to_hours(smc_metrics_cyc["expected_acrophase"])
arntl_h = smc_metrics_cyc.loc["Arntl", "acrophase_hours"] if "Arntl" in smc_metrics_cyc.index else 0
smc_metrics_cyc["acrophase_hours_bmal1_adj"] = (smc_metrics_cyc["acrophase_hours"] - arntl_h) % 24
print(f"  {len(smc_cyclers)} SMC cyclers")

inter_genes = sorted(set(smc_metrics_cyc.index) & set(jtk_df.index))
print(f"  {len(inter_genes)} overlapping with bulk")

# ── Step 3: Sample clock gene waveforms (major types) ─────────────────────

print("\nStep 3: Sampling clock gene waveforms for major cell types...")
major_clock_samples = {}
for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    alpha_df, beta_df, minmax_df = load_waveform_params(cluster, CONDITION)
    samples, genes = sample_waveforms(alpha_df, beta_df, minmax_df, CLOCK_GENES, NUM_WAVEFORM_SAMPLES)
    major_clock_samples[cell_type] = (samples, genes)
    print(f"  {cell_type}: {len(genes)}/{len(CLOCK_GENES)} clock genes available")

# ── Step 4: Sample clock gene waveforms (SMC subtypes) ────────────────────

print("\nStep 4: Sampling clock gene waveforms for SMC subtypes...")
smc_clock_samples = {}
for cluster, subtype_name in SMC_SUBTYPE_CLUSTERS.items():
    base = os.path.join(SMC_SUBCLUSTER_REG_DIR, cluster, CONDITION)
    alpha_df = pd.read_table(os.path.join(base, "gene_log_alpha.tsv"), sep="\t", index_col="gene")
    beta_df = pd.read_table(os.path.join(base, "gene_log_beta.tsv"), sep="\t", index_col="gene")
    minmax_df = pd.read_table(os.path.join(base, "log_min_max.tsv"), sep="\t", index_col="gene")
    samples, genes = sample_waveforms(alpha_df, beta_df, minmax_df, CLOCK_GENES, NUM_WAVEFORM_SAMPLES)
    smc_clock_samples[subtype_name] = (samples, genes)
    print(f"  {subtype_name}: {len(genes)}/{len(CLOCK_GENES)} clock genes available")

# ── Step 5: Load cycler acrophases for all major types ────────────────────

print("\nStep 5: Loading acrophase distributions for all cell types...")
cycler_acrophases = {}
for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    df = load_metrics(cluster, CONDITION)
    cyc = filter_cyclers(df)
    hours = [acrophase_rad_to_hours(df.loc[g, "expected_acrophase"]) for g in sorted(cyc)]
    cycler_acrophases[cell_type] = hours
    print(f"  {cell_type}: {len(hours)} cyclers")

# ── Step 6: Export source data ────────────────────────────────────────────

print("\nStep 6: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

scatter_df = pd.DataFrame({
    "gene": inter_genes,
    "bulk_acrophase_bmal1_adj": jtk_df.loc[inter_genes, "ADJ_LAG"].values,
    "sc_acrophase_bmal1_adj": smc_metrics_cyc.loc[inter_genes, "acrophase_hours_bmal1_adj"].values,
})
scatter_df.to_csv(os.path.join(SOURCE_DATA_DIR, "panel_a_bulk_vs_sc.csv"), index=False)

for cell_type, hours in cycler_acrophases.items():
    pd.DataFrame({"acrophase_hour": hours}).to_csv(
        os.path.join(SOURCE_DATA_DIR, f"panel_d_acrophase_{cell_type}.csv"), index=False
    )

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 7: Generate figure ──────────────────────────────────────────────

print("\nStep 7: Generating figure...")

from matplotlib.gridspec import GridSpecFromSubplotSpec

n_clock = len(CLOCK_GENES)
n_major = len(CLUSTER_CELL_TYPE)
n_smc = len(SMC_SUBTYPE_CLUSTERS)
cell_types_list = list(CLUSTER_CELL_TYPE.values())
smc_subtype_names = list(SMC_SUBTYPE_CLUSTERS.values())
smc_colors = plt.cm.Reds(np.linspace(0.3, 0.85, n_smc))
margin = 0.2

fig = plt.figure(figsize=(7.5, 12), dpi=600)
fig.patch.set_facecolor("white")

# Row 1: panels a (scatter) + d (histograms) side by side
# Row 2: panel b (major cell type violins)
# Row 3: panel c (SMC subtype violins)
gs_outer = GridSpec(
    3, 1, figure=fig, hspace=0.28,
    height_ratios=[1.0, n_major, n_smc],
    left=0.10, right=0.96, top=0.97, bottom=0.03,
)

# ── Row 1: Panel a (scatter) + Panel d (histograms) ──────────────────────

gs_ad = GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_outer[0], wspace=0.50)

# Panel a: scatter plot
ax_scatter = fig.add_subplot(gs_ad[0, 0])
ax_scatter.scatter(
    jtk_df.loc[inter_genes, "ADJ_LAG"],
    smc_metrics_cyc.loc[inter_genes, "acrophase_hours_bmal1_adj"],
    s=3, alpha=0.55, color=CELL_TYPE_COLORS["SMC"], edgecolors="none",
    zorder=3, rasterized=True,
)
ax_scatter.plot([0, 24], [0, 24], color="0.3", linewidth=0.7, linestyle="--", alpha=0.6, zorder=2)
ax_scatter.set_xlim(-0.5, 24.5)
ax_scatter.set_ylim(-0.5, 24.5)
ax_scatter.set_xticks([0, 6, 12, 18, 24])
ax_scatter.set_yticks([0, 6, 12, 18, 24])
ax_scatter.set_xlabel("Bulk aorta acrophase\n(hours, Bmal1-adj.)", fontsize=7)
ax_scatter.set_ylabel("Single-cell SMC acrophase\n(hours, Bmal1-adj.)", fontsize=7)
ax_scatter.set_title("SMC", fontsize=8, fontweight="semibold", color="0.15")
ax_scatter.text(0.05, 0.95, f"n = {len(inter_genes)}", transform=ax_scatter.transAxes,
                fontsize=6, ha="left", va="top", color="0.35")
ax_scatter.tick_params(labelsize=6)
sns.despine(ax=ax_scatter)
ax_scatter.set_aspect("equal")
ax_scatter.text(-0.30, 1.10, "a", transform=ax_scatter.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Panel d: acrophase histograms (on same row as scatter)
num_bins = 24
bin_edges = np.linspace(0, 24, num_bins + 1)
cell_types_for_hist = list(CLUSTER_CELL_TYPE.values())

for i, cell_type in enumerate(cell_types_for_hist):
    ax = fig.add_subplot(gs_ad[0, 1 + i])
    hours = cycler_acrophases[cell_type]
    ax.hist(hours, bins=bin_edges, color=CELL_TYPE_COLORS[cell_type],
            edgecolor="white", linewidth=0.3, alpha=0.85, rasterized=True)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["0", "6", "12", "18", "24"], fontsize=6)
    ax.set_xlabel("Acrophase (ZT hours)", fontsize=7)
    if i == 0:
        ax.set_ylabel("Number of genes", fontsize=7)
    ax.set_title(cell_type, fontsize=8, fontweight="semibold", color="0.15")
    ax.text(0.95, 0.93, f"n = {len(hours)}",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="top", color="0.4")
    sns.despine(ax=ax)
    ax.tick_params(axis="y", labelsize=6)
    if i == 0:
        ax.text(-0.25, 1.10, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# ── Helper: draw violin grid ─────────────────────────────────────────────

def _draw_violin_grid(gs_parent, row_labels, data_dict, colors_dict,
                      gene_min, gene_max, panel_label):
    n_rows = len(row_labels)
    gs_inner = GridSpecFromSubplotSpec(n_rows, n_clock, subplot_spec=gs_parent,
                                      hspace=0.35, wspace=0.35)
    for row_i, label in enumerate(row_labels):
        samples, genes = data_dict[label]
        color = colors_dict[label]
        for col_j, gene in enumerate(CLOCK_GENES):
            ax = fig.add_subplot(gs_inner[row_i, col_j])
            if gene in genes:
                idx = genes.index(gene)
                vals_log10 = samples[idx] / np.log(10)
                parts = ax.violinplot(
                    [vals_log10[:, t].flatten() for t in range(4)],
                    positions=[0, 6, 12, 18],
                    widths=3.2, showmeans=False, showmedians=True,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.55)
                    pc.set_edgecolor(color)
                    pc.set_linewidth(0.4)
                parts["cmedians"].set_color(color)
                parts["cmedians"].set_linewidth(0.6)

            ax.set_xticks([0, 6, 12, 18])
            ax.set_ylim(gene_min[col_j], gene_max[col_j])
            ax.tick_params(axis="y", labelsize=6)
            ax.tick_params(axis="x", labelsize=6)
            sns.despine(ax=ax)

            if row_i == n_rows - 1:
                ax.set_xticklabels(ZT_LABELS, fontsize=6)
            else:
                ax.set_xticklabels([])
            if row_i == 0:
                ax.set_title(gene, fontsize=8, fontweight="medium", style="italic")
            if col_j == 0:
                ax.set_ylabel(label, fontsize=7, fontweight="medium")
            else:
                ax.set_yticklabels([])

            if row_i == 0 and col_j == 0:
                ax.text(-0.40, 1.22, panel_label, transform=ax.transAxes,
                        fontsize=11, fontweight="bold", va="top")

# ── Panel c: Clock gene violins (major types) ────────────────────────────

gene_min_log10 = np.full(n_clock, np.inf)
gene_max_log10 = np.full(n_clock, -np.inf)
for cell_type in CLUSTER_CELL_TYPE.values():
    samples, genes = major_clock_samples[cell_type]
    for gi, gene in enumerate(CLOCK_GENES):
        if gene in genes:
            idx = genes.index(gene)
            vals_log10 = samples[idx] / np.log(10)
            gene_min_log10[gi] = min(gene_min_log10[gi], vals_log10.min())
            gene_max_log10[gi] = max(gene_max_log10[gi], vals_log10.max())
gene_min_log10 -= margin
gene_max_log10 += margin

major_colors = {ct: CELL_TYPE_COLORS[ct] for ct in cell_types_list}
_draw_violin_grid(gs_outer[1], cell_types_list, major_clock_samples,
                  major_colors, gene_min_log10, gene_max_log10, "c")

# ── Panel c: Clock gene violins (SMC subtypes) ───────────────────────────

smc_gene_min_log10 = np.full(n_clock, np.inf)
smc_gene_max_log10 = np.full(n_clock, -np.inf)
for subtype_name in SMC_SUBTYPE_CLUSTERS.values():
    samples, genes = smc_clock_samples[subtype_name]
    for gi, gene in enumerate(CLOCK_GENES):
        if gene in genes:
            idx = genes.index(gene)
            vals_log10 = samples[idx] / np.log(10)
            smc_gene_min_log10[gi] = min(smc_gene_min_log10[gi], vals_log10.min())
            smc_gene_max_log10[gi] = max(smc_gene_max_log10[gi], vals_log10.max())
smc_gene_min_log10 -= margin
smc_gene_max_log10 += margin

smc_color_dict = {name: smc_colors[i] for i, name in enumerate(smc_subtype_names)}
_draw_violin_grid(gs_outer[2], smc_subtype_names, smc_clock_samples,
                  smc_color_dict, smc_gene_min_log10, smc_gene_max_log10, "d")

# ── Save ──────────────────────────────────────────────────────────────────

os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_1_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=600, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
