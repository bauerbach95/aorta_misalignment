"""
Generate Figure 2: Cell-type-specific circadian rhythms in mouse aortic cell types.

Subplots:
  a) Bar chart: number of daily cyclers per major cell type (shared M/F)
  b) Rose plots: KEGG pathway acrophase enrichment (SMC, Fibroblast only)
  c) Rose plots: TF acrophase enrichment (SMC, Fibroblast only)

Output:
  figures/fig_2.pdf
  figures/fig_2_source_data/  (raw data tables for Nature submission)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import circmean

from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, CELL_TYPE_COLORS,
    FIGURES_DIR, ACROPHASE_HOUR_THRESH, TOP_N_PATHWAYS,
    load_metrics, filter_cyclers, acrophase_rad_to_hours, circular_hour_distance,
    load_kegg_dict, load_tf_dict, run_enrichment,
    make_rose_plot, export_enrichment_tables,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_2_source_data")
ROSE_PLOT_CELL_TYPES = ["SMC", "Fibroblast"]

# ── Step 1: Identify daily cyclers (shared M/F) ─────────────────────────────

print("Step 1: Identifying daily cyclers shared between male and female...")

daily_cyclers = {}
daily_cycler_acrophases = {}
cycler_details = {}

for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    male_df = load_metrics(cluster, ALIGNED_CONDITIONS["male"])
    female_df = load_metrics(cluster, ALIGNED_CONDITIONS["female"])

    shared = filter_cyclers(male_df) & filter_cyclers(female_df)
    agreed = set()
    acrophases = {}
    details_rows = []
    for gene in sorted(shared):
        m_acro = male_df.loc[gene, "expected_acrophase"]
        f_acro = female_df.loc[gene, "expected_acrophase"]
        m_h = acrophase_rad_to_hours(m_acro)
        f_h = acrophase_rad_to_hours(f_acro)
        dist = circular_hour_distance(m_h, f_h)
        if dist <= ACROPHASE_HOUR_THRESH:
            agreed.add(gene)
            avg_acro = circmean([m_acro, f_acro], high=2 * np.pi, low=0)
            acrophases[gene] = avg_acro
            details_rows.append({
                "gene": gene,
                "male_acrophase_hour": round(m_h, 2),
                "female_acrophase_hour": round(f_h, 2),
                "circular_distance_hour": round(float(dist), 2),
                "average_acrophase_hour": round(acrophase_rad_to_hours(avg_acro), 2),
                "average_acrophase_rad": round(float(avg_acro), 4),
                "male_bf_log10": round(male_df.loc[gene, "waveform_over_circadian_component_subtracted_log10_bf"], 2),
                "female_bf_log10": round(female_df.loc[gene, "waveform_over_circadian_component_subtracted_log10_bf"], 2),
            })

    daily_cyclers[cell_type] = agreed
    daily_cycler_acrophases[cell_type] = acrophases
    cycler_details[cell_type] = pd.DataFrame(details_rows)
    print(f"  {cell_type}: {len(agreed)} daily cyclers")

# ── Step 2: KEGG pathway acrophase enrichment ────────────────────────────────

print("\nStep 2: KEGG pathway acrophase enrichment...")
kegg_dict = load_kegg_dict()

kegg_enrichments = {}
kegg_enrichments_full = {}
for cell_type in CLUSTER_CELL_TYPE.values():
    full_df, top_df, lrt_df = run_enrichment(
        daily_cycler_acrophases[cell_type], kegg_dict, cell_type
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
        daily_cycler_acrophases[cell_type], tf_dict, cell_type
    )
    tf_enrichments_full[cell_type] = (full_df, lrt_df)
    tf_enrichments[cell_type] = (top_df, lrt_df)

# ── Step 4: Export source data ───────────────────────────────────────────────

print("\nStep 4: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

bar_data = pd.DataFrame({
    "cell_type": list(CLUSTER_CELL_TYPE.values()),
    "num_daily_cyclers": [len(daily_cyclers[ct]) for ct in CLUSTER_CELL_TYPE.values()],
})
bar_data.to_csv(os.path.join(SOURCE_DATA_DIR, "panel_a_daily_cycler_counts.csv"), index=False)

for ct in CLUSTER_CELL_TYPE.values():
    if not cycler_details[ct].empty:
        cycler_details[ct].to_csv(
            os.path.join(SOURCE_DATA_DIR, f"daily_cyclers_{ct}.csv"), index=False
        )

export_enrichment_tables(SOURCE_DATA_DIR, "panel_b_kegg", kegg_enrichments_full, "pathway")
export_enrichment_tables(SOURCE_DATA_DIR, "panel_c_tf", tf_enrichments_full, "tf_motif")
print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 5: Generate Figure 2 ───────────────────────────────────────────────

print("\nStep 5: Generating figure...")

all_cell_types = ["SMC", "Fibroblast", "EC", "Macrophage"]
n_rose = len(ROSE_PLOT_CELL_TYPES)

fig = plt.figure(figsize=(8.5, 11), dpi=300)
fig.patch.set_facecolor("white")

gs = GridSpec(
    3, n_rose, figure=fig, hspace=0.75, wspace=0.65,
    height_ratios=[0.5, 1.2, 1.2],
    left=0.08, right=0.92, top=0.95, bottom=0.04,
)

# Row a: Bar chart
ax_bar = fig.add_subplot(gs[0, :])
x = np.arange(len(all_cell_types))
daily_counts = [len(daily_cyclers[ct]) for ct in all_cell_types]
bar_colors = [CELL_TYPE_COLORS[ct] for ct in all_cell_types]

bars = ax_bar.bar(x, daily_counts, width=0.55, color=bar_colors,
                  edgecolor="white", linewidth=0.8, zorder=3)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(all_cell_types, fontsize=7, rotation=0, ha="center")
ax_bar.set_ylabel("Daily cyclers\n(shared M & F)", fontsize=7)
ax_bar.set_xlim(-0.6, len(all_cell_types) - 0.4)
ax_bar.tick_params(axis="y", labelsize=6)
sns.despine(ax=ax_bar, bottom=True)
ax_bar.tick_params(axis="x", length=0)
ax_bar.set_axisbelow(True)
ax_bar.yaxis.grid(True, alpha=0.15, linewidth=0.4)

for bar in bars:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h + max(daily_counts) * 0.02,
                str(int(h)), ha="center", va="bottom", fontsize=6.5, fontweight="medium",
                color="0.25")

ax_bar.text(-0.06, 1.08, "a", transform=ax_bar.transAxes,
            fontsize=11, fontweight="bold", va="top")

# Row b: KEGG rose plots
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
    ax = fig.add_subplot(gs[1, i], projection="polar")
    enrich_df, lrt_df = kegg_enrichments[ct]
    make_rose_plot(ax, enrich_df, lrt_df, ct, is_tf=False)
    if i == 0:
        ax.text(-0.25, 1.22, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.62, "KEGG Pathways", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Row c: TF rose plots
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
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
outpath = os.path.join(FIGURES_DIR, "fig_2.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
