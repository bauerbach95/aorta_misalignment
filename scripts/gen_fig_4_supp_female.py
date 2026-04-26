"""
Generate Supplementary Figure: Female misalignment effects.

Simplified version of Figure 4 for female mice:
  a) Amplitude loss histograms — female aligned vs female misaligned (4 cell types)
  b) Acrophase shift scatter — female aligned vs female misaligned (4 cell types)

Output:
  figures/fig_4_supp_female.pdf
  figures/fig_4_supp_female_source_data/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, MISALIGNED_CONDITIONS,
    CELL_TYPE_COLORS, FIGURES_DIR,
    load_metrics, filter_cyclers, load_waveform_params, sample_waveforms,
    compute_amplitude, acrophase_rad_to_hours,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_4_supp_female_source_data")
NUM_SAMPLES = 5000
SEX = "female"
COND_AL = ALIGNED_CONDITIONS[SEX]
COND_MIS = MISALIGNED_CONDITIONS[SEX]

all_cell_types = list(CLUSTER_CELL_TYPE.values())

# ── Step 1: Panel a — amplitude loss ────────────────────────────────────────

print("Step 1: Amplitude loss under misalignment (female)...")

amp_fracs = {}

for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    metrics_al = load_metrics(cluster, COND_AL)
    metrics_mis = load_metrics(cluster, COND_MIS)
    cyclers_al = filter_cyclers(metrics_al)

    if len(cyclers_al) < 5:
        print(f"  {cell_type}: SKIP ({len(cyclers_al)} aligned cyclers)")
        amp_fracs[cell_type] = np.array([])
        continue

    alpha_al, beta_al, mm_al = load_waveform_params(cluster, COND_AL)
    alpha_mis, beta_mis, mm_mis = load_waveform_params(cluster, COND_MIS)

    gene_list = sorted(cyclers_al)
    samp_al, genes_al = sample_waveforms(alpha_al, beta_al, mm_al, gene_list, NUM_SAMPLES)
    samp_mis, genes_mis = sample_waveforms(alpha_mis, beta_mis, mm_mis, gene_list, NUM_SAMPLES)

    idx_al = {g: i for i, g in enumerate(genes_al)}
    idx_mis = {g: i for i, g in enumerate(genes_mis)}
    common = [g for g in gene_list if g in idx_al and g in idx_mis]
    ord_al = [idx_al[g] for g in common]
    ord_mis = [idx_mis[g] for g in common]

    amp_al = compute_amplitude(samp_al[ord_al])
    amp_mis = compute_amplitude(samp_mis[ord_mis])
    frac_mis_larger = (amp_mis > amp_al).mean(axis=1)
    amp_fracs[cell_type] = frac_mis_larger
    print(f"  {cell_type}: {len(common)} genes, "
          f"median P(mis>al) = {np.median(frac_mis_larger):.3f}")

# ── Step 2: Panel b — acrophase shifts ──────────────────────────────────────

print("\nStep 2: Acrophase shifts under misalignment (female)...")

acro_data = {}

for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    metrics_al = load_metrics(cluster, COND_AL)
    metrics_mis = load_metrics(cluster, COND_MIS)
    cyclers_al = filter_cyclers(metrics_al)
    cyclers_mis = filter_cyclers(metrics_mis)
    shared = sorted(cyclers_al & cyclers_mis)

    if len(shared) < 3:
        acro_data[cell_type] = (np.array([]), np.array([]), [])
        print(f"  {cell_type}: {len(shared)} shared cyclers")
        continue

    acro_al_h = np.array([acrophase_rad_to_hours(metrics_al.loc[g, "expected_acrophase"])
                          for g in shared])
    acro_mis_h = np.array([acrophase_rad_to_hours(metrics_mis.loc[g, "expected_acrophase"])
                           for g in shared])
    acro_data[cell_type] = (acro_al_h, acro_mis_h, shared)
    print(f"  {cell_type}: {len(shared)} shared cyclers")

# ── Step 3: Export source data ──────────────────────────────────────────────

print("\nStep 3: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

for ct, fracs in amp_fracs.items():
    if len(fracs) > 0:
        pd.DataFrame({"frac_misaligned_amplitude_larger": fracs}).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_a_amplitude_fracs_{ct}.csv"), index=False)

for ct, (acro_al, acro_mis, genes) in acro_data.items():
    if len(genes) > 0:
        pd.DataFrame({"gene": genes, "acrophase_aligned_h": acro_al,
                       "acrophase_misaligned_h": acro_mis}).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_b_acrophase_{ct}.csv"), index=False)

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 4: Generate figure ─────────────────────────────────────────────────

print("\nStep 4: Generating figure...")

fig = plt.figure(figsize=(8.5, 5.5), dpi=300)
fig.patch.set_facecolor("white")

gs_main = GridSpec(2, 1, figure=fig, hspace=0.5,
                   height_ratios=[1, 1],
                   left=0.08, right=0.95, top=0.93, bottom=0.08)

# ── Row a: amplitude loss histograms ────────────────────────────────────────

gs_a = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], wspace=0.35)

for i, ct in enumerate(all_cell_types):
    ax = fig.add_subplot(gs_a[0, i])
    fracs = amp_fracs[ct]
    if len(fracs) > 0:
        ax.hist(fracs, bins=25, color=CELL_TYPE_COLORS[ct], edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.axvline(0.5, color="k", linestyle="--", linewidth=0.6, alpha=0.5)
        med = np.median(fracs)
        ax.text(0.97, 0.95, f"n={len(fracs)}\nmed={med:.2f}",
                transform=ax.transAxes, fontsize=5, ha="right", va="top", color="0.35")
    ax.set_title(ct, fontsize=7, fontweight="semibold", color=CELL_TYPE_COLORS[ct])
    ax.set_xlabel("Fraction amplitude\nlarger in misaligned" if i == 1 else "", fontsize=5.5)
    ax.set_ylabel("Number of genes" if i == 0 else "", fontsize=6)
    ax.set_xlim(0, 1)
    sns.despine(ax=ax)
    if i == 0:
        ax.text(-0.2, 1.15, "a", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# ── Row b: acrophase scatter ────────────────────────────────────────────────

gs_b = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[1], wspace=0.35)

for i, ct in enumerate(all_cell_types):
    ax = fig.add_subplot(gs_b[0, i])
    acro_al, acro_mis, genes = acro_data[ct]
    if len(genes) > 0:
        ax.scatter(acro_al, acro_mis, s=8, alpha=0.5, color=CELL_TYPE_COLORS[ct],
                   edgecolors="none")
        ax.plot([0, 24], [0, 24], "k--", linewidth=0.5, alpha=0.4)
        ax.text(0.97, 0.05, f"n={len(genes)}", transform=ax.transAxes,
                fontsize=5, ha="right", va="bottom", color="0.35")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_aspect("equal")
    ax.set_title(ct, fontsize=7, fontweight="semibold", color=CELL_TYPE_COLORS[ct])
    ax.set_xlabel("Aligned acrophase (h)" if i == 1 else "", fontsize=5.5)
    ax.set_ylabel("Misaligned\nacrophase (h)" if i == 0 else "", fontsize=6)
    sns.despine(ax=ax)
    if i == 0:
        ax.text(-0.2, 1.12, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_4_supp_female.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
