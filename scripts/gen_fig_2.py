"""
Generate Figure 2: Cell-type-specific circadian rhythms in mouse aortic cell types.

Subplots:
  a) Bar chart: number of daily cyclers per major cell type (shared M/F)
  b) Rose plots: Reactome pathway acrophase enrichment (SMC, Fibroblast only)
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
    load_reactome_dict, clean_reactome_name, load_tf_dict, run_enrichment,
    make_rose_plot, export_enrichment_tables,
    load_smc_switching_genes,
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

# ── Step 2: Reactome pathway acrophase enrichment ────────────────────────────

print("\nStep 2: Reactome pathway acrophase enrichment...")
reactome_dict = load_reactome_dict()

reactome_enrichments = {}
reactome_enrichments_full = {}
for cell_type in CLUSTER_CELL_TYPE.values():
    full_df, top_df, lrt_df = run_enrichment(
        daily_cycler_acrophases[cell_type], reactome_dict, cell_type
    )
    reactome_enrichments_full[cell_type] = (full_df, lrt_df)
    reactome_enrichments[cell_type] = (top_df, lrt_df)

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

# ── Step 3b: SMC phenotypic switching acrophase analysis ─────────────────────

print("\nStep 3b: SMC phenotypic switching acrophase analysis...")
switching_genes = load_smc_switching_genes()

shared_smc_acro = daily_cycler_acrophases["SMC"]
shared_smc_cycler_list = list(shared_smc_acro.keys())

switch_up_acro = {g: shared_smc_acro[g] for g in switching_genes["up"] if g in shared_smc_acro}
switch_down_acro = {g: shared_smc_acro[g] for g in switching_genes["down"] if g in shared_smc_acro}
print(f"  Shared M/F SMC cyclers: {len(shared_smc_cycler_list)}")
print(f"  Switching UP genes among shared cyclers: {len(switch_up_acro)}/{len(switching_genes['up'])}")
print(f"  Switching DOWN genes among shared cyclers: {len(switch_down_acro)}/{len(switching_genes['down'])}")

NUM_PERMUTATIONS = 5000
shared_acro_series = pd.Series(shared_smc_acro)

DUSK_LO, DUSK_HI = 2.0, 4.0  # radians — roughly ZT8–ZT16
DAWN_LO, DAWN_HI = (20 / 12) * np.pi, (4 / 12) * np.pi  # wraps around 0


def count_in_window(acrophases, lo, hi, wraps=False):
    a = np.asarray(acrophases)
    if wraps:
        return int(np.sum((a >= lo) | (a <= hi)))
    return int(np.sum((a >= lo) & (a <= hi)))


def window_permutation_pvalue(observed_acro, all_cyclers, acro_series,
                              lo, hi, wraps, n_perm):
    obs = count_in_window(list(observed_acro.values()), lo, hi, wraps)
    n = len(observed_acro)
    count_ge = 0
    for _ in range(n_perm):
        perm = np.random.choice(all_cyclers, size=n, replace=True)
        perm_stat = count_in_window(acro_series.loc[perm].values, lo, hi, wraps)
        if perm_stat >= obs:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1), obs


switch_up_pval, switch_up_count = window_permutation_pvalue(
    switch_up_acro, shared_smc_cycler_list, shared_acro_series,
    DUSK_LO, DUSK_HI, wraps=False, n_perm=NUM_PERMUTATIONS)
switch_down_pval, switch_down_count = window_permutation_pvalue(
    switch_down_acro, shared_smc_cycler_list, shared_acro_series,
    DAWN_LO, DAWN_HI, wraps=True, n_perm=NUM_PERMUTATIONS)
print(f"  Dusk window [{DUSK_LO:.2f}, {DUSK_HI:.2f}] rad: "
      f"{switch_up_count}/{len(switch_up_acro)} UP genes, p={switch_up_pval:.4f}")
print(f"  Dawn window [>={DAWN_LO:.2f} or <={DAWN_HI:.2f}] rad: "
      f"{switch_down_count}/{len(switch_down_acro)} DOWN genes, p={switch_down_pval:.4f}")

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

export_enrichment_tables(SOURCE_DATA_DIR, "panel_b_reactome", reactome_enrichments_full, "pathway")
export_enrichment_tables(SOURCE_DATA_DIR, "panel_c_tf", tf_enrichments_full, "tf_motif")

# Panel d source data
for direction, acro_dict, pval in [
    ("up", switch_up_acro, switch_up_pval),
    ("down", switch_down_acro, switch_down_pval),
]:
    rows = [{"gene": g, "acrophase_rad": float(a),
             "acrophase_hour": round(acrophase_rad_to_hours(a), 2)}
            for g, a in sorted(acro_dict.items())]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_d_switching_{direction}.csv"), index=False)
pd.DataFrame([{"direction": "up", "n_genes": len(switch_up_acro),
                "permutation_pvalue": switch_up_pval},
               {"direction": "down", "n_genes": len(switch_down_acro),
                "permutation_pvalue": switch_down_pval}]).to_csv(
    os.path.join(SOURCE_DATA_DIR, "panel_d_switching_permutation_test.csv"), index=False)
print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 5: Generate Figure 2 ───────────────────────────────────────────────

print("\nStep 5: Generating figure...")

all_cell_types = ["SMC", "Fibroblast", "EC", "Macrophage"]
n_rose = len(ROSE_PLOT_CELL_TYPES)

fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
fig.patch.set_facecolor("white")

gs = GridSpec(
    4, n_rose, figure=fig, hspace=0.75, wspace=0.65,
    height_ratios=[0.5, 1.2, 1.2, 0.55],
    left=0.08, right=0.92, top=0.96, bottom=0.03,
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

# Row b: Reactome rose plots
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
    ax = fig.add_subplot(gs[1, i], projection="polar")
    enrich_df, lrt_df = reactome_enrichments[ct]
    # Clean Reactome names for display
    if not enrich_df.empty:
        cleaned_df = enrich_df.copy()
        cleaned_df.index = [clean_reactome_name(n) for n in cleaned_df.index]
        cleaned_lrt = lrt_df.loc[enrich_df.index].copy()
        cleaned_lrt.index = cleaned_df.index
    else:
        cleaned_df, cleaned_lrt = enrich_df, lrt_df
    make_rose_plot(ax, cleaned_df, cleaned_lrt, ct, is_tf=False)
    if i == 0:
        ax.text(-0.25, 1.22, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.62, "Reactome\nPathways", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Row c: TF rose plots
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
    ax = fig.add_subplot(gs[2, i], projection="polar")
    enrich_df, lrt_df = tf_enrichments[ct]
    make_rose_plot(ax, enrich_df, lrt_df, ct, is_tf=True)
    if i == 0:
        ax.text(-0.25, 1.22, "c", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.28, "TF Motifs", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Row d: SMC phenotypic switching histograms
num_bins_hist = 12
bin_edges_h = np.linspace(0, 24, num_bins_hist + 1)

for i, (direction, acro_dict, pval, title) in enumerate([
    ("up", switch_up_acro, switch_up_pval, "Increasing with\nphenotypic switching"),
    ("down", switch_down_acro, switch_down_pval, "Decreasing with\nphenotypic switching"),
]):
    ax = fig.add_subplot(gs[3, i])
    hours = [acrophase_rad_to_hours(a) for a in acro_dict.values()]
    ax.hist(hours, bins=bin_edges_h, color=CELL_TYPE_COLORS["SMC"],
            edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["ZT0", "ZT6", "ZT12", "ZT18", "ZT24"], fontsize=5.5)
    ax.set_xlabel("Acrophase (hours)", fontsize=6.5)
    ax.set_ylabel("Number of genes" if i == 0 else "", fontsize=6.5)
    ax.set_title(title, fontsize=7, fontweight="medium", color="0.2")
    ax.text(0.97, 0.95, f"n={len(acro_dict)}\np={pval:.3f}",
            transform=ax.transAxes, fontsize=5, ha="right", va="top", color="0.35")
    sns.despine(ax=ax)
    if i == 0:
        ax.text(-0.15, 1.15, "d", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.09, "SMC Switching", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_2.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
