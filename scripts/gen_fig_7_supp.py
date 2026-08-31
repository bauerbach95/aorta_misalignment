"""
Generate Supplementary Figure 7: iKO validation.

Panel a: Western blot of Bmal1 protein (from supp_fig_8_bmal1_ko_western_blot.pdf)
Panel b: Core clock gene waveforms in male Bmal1 KO mice across 4 major cell types

Output:
  figures/fig_7_supp.pdf
  figures/fig_7_supp_source_data/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, CELL_TYPE_COLORS,
    FIGURES_DIR, NONPARAM_REG_DIR,
    load_waveform_params, sample_waveforms, ZT_LABELS,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_7_supp_source_data")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

NUM_WAVEFORM_SAMPLES = 300
CLOCK_GENES = ["Arntl", "Nfil3", "Dbp", "Nr1d2"]
CONDITION = "male aligned bmal1-ko"
WESTERN_BLOT_PDF = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "old_figures", "supp_fig_8_bmal1_ko_western_blot.pdf",
)

n_clock = len(CLOCK_GENES)
cell_types_list = list(CLUSTER_CELL_TYPE.values())
n_types = len(cell_types_list)

# ── Convert western blot PDF to image ────────────────────────────────────

print("Converting western blot PDF to image...")
wb_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
subprocess.run(
    ["sips", "-s", "format", "png", "--out", wb_png,
     "-s", "dpiHeight", "300", "-s", "dpiWidth", "300",
     WESTERN_BLOT_PDF],
    capture_output=True,
)
wb_img = np.array(Image.open(wb_png))
os.unlink(wb_png)

# ── Sample clock gene waveforms in KO ────────────────────────────────────

print("Sampling clock gene waveforms in male KO...")
ko_samples = {}
for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    alpha_df, beta_df, minmax_df = load_waveform_params(cluster, CONDITION)
    samples, genes = sample_waveforms(alpha_df, beta_df, minmax_df, CLOCK_GENES, NUM_WAVEFORM_SAMPLES)
    ko_samples[cell_type] = (samples, genes)
    print(f"  {cell_type}: {len(genes)}/{n_clock} clock genes available")

# ── Export source data ───────────────────────────────────────────────────

for cell_type in cell_types_list:
    samples, genes = ko_samples[cell_type]
    for gi, gene in enumerate(genes):
        vals_log10 = samples[gi] / np.log(10)
        rows = []
        for ti, zt in enumerate([0, 6, 12, 18]):
            for si in range(vals_log10.shape[0]):
                rows.append({"ZT": zt, "sample": si, "log10_prop": vals_log10[si, ti]})
        pd.DataFrame(rows).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_b_{cell_type}_{gene}.csv"), index=False
        )
print(f"Source data written to {SOURCE_DATA_DIR}/")

# ── Compute y-axis limits ────────────────────────────────────────────────

gene_min_log10 = np.full(n_clock, np.inf)
gene_max_log10 = np.full(n_clock, -np.inf)
margin = 0.2

for cell_type in cell_types_list:
    samples, genes = ko_samples[cell_type]
    for gi, gene in enumerate(CLOCK_GENES):
        if gene in genes:
            idx = genes.index(gene)
            vals_log10 = samples[idx] / np.log(10)
            gene_min_log10[gi] = min(gene_min_log10[gi], vals_log10.min())
            gene_max_log10[gi] = max(gene_max_log10[gi], vals_log10.max())
gene_min_log10 -= margin
gene_max_log10 += margin

# ── Generate figure ──────────────────────────────────────────────────────

print("Generating figure...")

fig = plt.figure(figsize=(7.0, 8.5), dpi=300)
fig.patch.set_facecolor("white")

gs_outer = GridSpec(2, 1, figure=fig, hspace=0.25,
                    height_ratios=[1.0, 1.2],
                    left=0.10, right=0.96, top=0.97, bottom=0.05)

# ── Panel a: Western blot ────────────────────────────────────────────────

ax_wb = fig.add_subplot(gs_outer[0])
ax_wb.imshow(wb_img, aspect="equal")
ax_wb.axis("off")
ax_wb.text(-0.02, 1.02, "a", transform=ax_wb.transAxes,
           fontsize=11, fontweight="bold", va="top")

# ── Panel b: Clock gene violin grid ─────────────────────────────────────

gs_violins = GridSpecFromSubplotSpec(n_types, n_clock, subplot_spec=gs_outer[1],
                                     hspace=0.4, wspace=0.35)

for row_i, cell_type in enumerate(cell_types_list):
    samples, genes = ko_samples[cell_type]
    color = CELL_TYPE_COLORS[cell_type]
    for col_j, gene in enumerate(CLOCK_GENES):
        ax = fig.add_subplot(gs_violins[row_i, col_j])
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
        ax.set_ylim(gene_min_log10[col_j], gene_max_log10[col_j])
        ax.tick_params(axis="y", labelsize=5)
        ax.tick_params(axis="x", labelsize=5)
        sns.despine(ax=ax)

        if row_i == n_types - 1:
            ax.set_xticklabels(ZT_LABELS, fontsize=5)
            ax.set_xlabel("ZT", fontsize=6)
        else:
            ax.set_xticklabels([])
        if row_i == 0:
            ax.set_title(gene, fontsize=7, fontweight="medium", style="italic")
        if col_j == 0:
            ax.set_ylabel(f"{cell_type}\nLog10 prop.", fontsize=6, fontweight="medium")
            if row_i == 0:
                ax.text(-0.40, 1.22, "b", transform=ax.transAxes,
                        fontsize=11, fontweight="bold", va="top")
        else:
            ax.set_yticklabels([])

outpath = os.path.join(FIGURES_DIR, "fig_7_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"Figure saved to {outpath}")
