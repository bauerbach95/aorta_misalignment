"""
Supplementary figure: Female-specific circadian cyclers in mouse aortic cell types.

Subplots:
  a) Bar chart: number of female cyclers per major cell type
  b) Rose plots: KEGG pathway acrophase enrichment (cell types with sufficient cyclers)
  c) Rose plots: TF acrophase enrichment (cell types with sufficient cyclers)

Output:
  figures/section_2_supp_female_cyclers.pdf
  figures/section_2_supp_female_cyclers_source_data/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, CELL_TYPE_COLORS,
    FIGURES_DIR,
    load_metrics, filter_cyclers, acrophase_rad_to_hours,
    load_kegg_dict, load_tf_dict, run_enrichment,
    make_rose_plot, export_enrichment_tables,
)

apply_style()

SEX = "female"
FIGURE_NAME = "section_2_supp_female_cyclers"
SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, f"{FIGURE_NAME}_source_data")
MIN_CYCLERS_FOR_ROSE = 15

# ── Step 1: Identify male cyclers ────────────────────────────────────────────

print(f"Step 1: Identifying {SEX} cyclers...")

cyclers = {}
cycler_acrophases = {}
cycler_details = {}

for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    df = load_metrics(cluster, ALIGNED_CONDITIONS[SEX])
    cyc = filter_cyclers(df)

    acrophases = {}
    details_rows = []
    for gene in sorted(cyc):
        acro = df.loc[gene, "expected_acrophase"]
        h = acrophase_rad_to_hours(acro)
        acrophases[gene] = acro
        details_rows.append({
            "gene": gene,
            "acrophase_hour": round(h, 2),
            "acrophase_rad": round(float(acro), 4),
            "bf_log10": round(df.loc[gene, "waveform_over_circadian_component_subtracted_log10_bf"], 2),
            "expected_mesor": round(df.loc[gene, "expected_mesor"], 4),
        })

    cyclers[cell_type] = cyc
    cycler_acrophases[cell_type] = acrophases
    cycler_details[cell_type] = pd.DataFrame(details_rows)
    print(f"  {cell_type}: {len(cyc)} cyclers")

rose_cell_types = [ct for ct in CLUSTER_CELL_TYPE.values() if len(cyclers[ct]) >= MIN_CYCLERS_FOR_ROSE]
print(f"  Rose plots for: {rose_cell_types}")

# ── Step 2: KEGG pathway acrophase enrichment ────────────────────────────────

print("\nStep 2: KEGG pathway acrophase enrichment...")
kegg_dict = load_kegg_dict()

kegg_enrichments = {}
kegg_enrichments_full = {}
for cell_type in CLUSTER_CELL_TYPE.values():
    full_df, top_df, lrt_df = run_enrichment(
        cycler_acrophases[cell_type], kegg_dict, cell_type
    )
    kegg_enrichments_full[cell_type] = (full_df, lrt_df)
    kegg_enrichments[cell_type] = (top_df, lrt_df)

# ── Step 3: TF acrophase enrichment ──────────────────────────────────────────

print("\nStep 3: TF acrophase enrichment...")
tf_dict = load_tf_dict()

tf_enrichments = {}
tf_enrichments_full = {}
for cell_type in CLUSTER_CELL_TYPE.values():
    full_df, top_df, lrt_df = run_enrichment(
        cycler_acrophases[cell_type], tf_dict, cell_type
    )
    tf_enrichments_full[cell_type] = (full_df, lrt_df)
    tf_enrichments[cell_type] = (top_df, lrt_df)

# ── Step 4: Export source data ───────────────────────────────────────────────

print("\nStep 4: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

bar_data = pd.DataFrame({
    "cell_type": list(CLUSTER_CELL_TYPE.values()),
    f"num_{SEX}_cyclers": [len(cyclers[ct]) for ct in CLUSTER_CELL_TYPE.values()],
})
bar_data.to_csv(os.path.join(SOURCE_DATA_DIR, "panel_a_cycler_counts.csv"), index=False)

for ct in CLUSTER_CELL_TYPE.values():
    if not cycler_details[ct].empty:
        cycler_details[ct].to_csv(
            os.path.join(SOURCE_DATA_DIR, f"cyclers_{ct}.csv"), index=False
        )

export_enrichment_tables(SOURCE_DATA_DIR, "panel_b_kegg", kegg_enrichments_full, "pathway")
export_enrichment_tables(SOURCE_DATA_DIR, "panel_c_tf", tf_enrichments_full, "tf_motif")
print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 5: Generate figure ──────────────────────────────────────────────────

print("\nStep 5: Generating figure...")

all_cell_types = list(CLUSTER_CELL_TYPE.values())
n_rose = len(rose_cell_types)

fig = plt.figure(figsize=(8.5, 11), dpi=300)
fig.patch.set_facecolor("white")

gs = GridSpec(
    3, max(n_rose, 1), figure=fig, hspace=0.75, wspace=0.65,
    height_ratios=[0.5, 1.2, 1.2],
    left=0.08, right=0.92, top=0.95, bottom=0.04,
)

# Row a: Bar chart
ax_bar = fig.add_subplot(gs[0, :])
x = np.arange(len(all_cell_types))
counts = [len(cyclers[ct]) for ct in all_cell_types]
bar_colors = [CELL_TYPE_COLORS[ct] for ct in all_cell_types]

bars = ax_bar.bar(x, counts, width=0.55, color=bar_colors,
                  edgecolor="white", linewidth=0.8, zorder=3)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(all_cell_types, fontsize=7, rotation=0, ha="center")
ax_bar.set_ylabel(f"Daily cyclers\n({SEX})", fontsize=7)
ax_bar.set_xlim(-0.6, len(all_cell_types) - 0.4)
ax_bar.tick_params(axis="y", labelsize=6)
sns.despine(ax=ax_bar, bottom=True)
ax_bar.tick_params(axis="x", length=0)
ax_bar.set_axisbelow(True)
ax_bar.yaxis.grid(True, alpha=0.15, linewidth=0.4)

for bar in bars:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.02,
                str(int(h)), ha="center", va="bottom", fontsize=6.5, fontweight="medium",
                color="0.25")

ax_bar.text(-0.06, 1.08, "a", transform=ax_bar.transAxes,
            fontsize=11, fontweight="bold", va="top")

# Row b: KEGG rose plots
for i, ct in enumerate(rose_cell_types):
    ax = fig.add_subplot(gs[1, i], projection="polar")
    enrich_df, lrt_df = kegg_enrichments[ct]
    make_rose_plot(ax, enrich_df, lrt_df, ct, is_tf=False)
    if i == 0:
        ax.text(-0.25, 1.22, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.62, "KEGG Pathways", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Row c: TF rose plots
for i, ct in enumerate(rose_cell_types):
    ax = fig.add_subplot(gs[2, i], projection="polar")
    enrich_df, lrt_df = tf_enrichments[ct]
    make_rose_plot(ax, enrich_df, lrt_df, ct, is_tf=True)
    if i == 0:
        ax.text(-0.25, 1.22, "c", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.24, "TF Motifs", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, f"{FIGURE_NAME}.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
