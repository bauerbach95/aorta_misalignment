"""
Generate Supplementary Figure 2: KEGG and GO:BP acrophase enrichment.

Extends Figure 2b (Reactome) with additional gene set databases for shared M/F
daily cyclers in SMC and Fibroblast.

Subplots:
  a) Rose plots: KEGG pathway acrophase enrichment (SMC, Fibroblast)
  b) Rose plots: GO Biological Process acrophase enrichment (SMC, Fibroblast)

Output:
  figures/fig_2_supp.pdf
  figures/fig_2_supp_source_data/
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import circmean

from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, CELL_TYPE_COLORS,
    FIGURES_DIR, ACROPHASE_HOUR_THRESH, TOP_N_PATHWAYS, GENE_SETS_DIR,
    load_metrics, filter_cyclers, acrophase_rad_to_hours, circular_hour_distance,
    load_kegg_dict, parse_gsea_set_file, run_enrichment,
    make_rose_plot, export_enrichment_tables,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_2_supp_source_data")
ROSE_PLOT_CELL_TYPES = ["SMC", "Fibroblast"]

GO_EXCLUDE_PATTERNS = [
    r"VIRUS", r"VIRAL", r"BACTERIUM", r"BACTERIAL", r"PARASITE", r"PARASIT",
    r"SYMBIONT", r"SYMBIOSIS",
    r"OLFACT", r"TASTE", r"PHEROMONE",
    r"EMBRYONIC", r"EMBRYO_DEVELOPMENT",
    r"SPERMAT", r"OOCYTE", r"OVULAT",
    r"NEURON_MIGRATION", r"NEUROGENESIS",
]


def filter_gobp_dict(d, max_genes=300):
    pattern = re.compile("|".join(GO_EXCLUDE_PATTERNS), re.IGNORECASE)
    return {k: v for k, v in d.items() if not pattern.search(k) and len(v) <= max_genes}


def clean_gobp_name(name):
    """GOBP_SMOOTH_MUSCLE_CONTRACTION -> Smooth muscle contraction."""
    name = re.sub(r"^GOBP_", "", name)
    name = name.replace("_", " ")
    return name[0].upper() + name[1:].lower() if name else name


# ── Step 1: Identify daily cyclers (shared M/F) ─────────────────────────────

print("Step 1: Identifying daily cyclers shared between male and female...")

daily_cycler_acrophases = {}

for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    male_df = load_metrics(cluster, ALIGNED_CONDITIONS["male"])
    female_df = load_metrics(cluster, ALIGNED_CONDITIONS["female"])

    shared = filter_cyclers(male_df) & filter_cyclers(female_df)
    acrophases = {}
    for gene in sorted(shared):
        m_acro = male_df.loc[gene, "expected_acrophase"]
        f_acro = female_df.loc[gene, "expected_acrophase"]
        m_h = acrophase_rad_to_hours(m_acro)
        f_h = acrophase_rad_to_hours(f_acro)
        dist = circular_hour_distance(m_h, f_h)
        if dist <= ACROPHASE_HOUR_THRESH:
            avg_acro = circmean([m_acro, f_acro], high=2 * np.pi, low=0)
            acrophases[gene] = avg_acro

    daily_cycler_acrophases[cell_type] = acrophases
    print(f"  {cell_type}: {len(acrophases)} daily cyclers")

# ── Step 2: Load gene set databases ──────────────────────────────────────────

print("\nStep 2: Loading gene set databases...")

kegg_dict = load_kegg_dict()

gobp_raw = parse_gsea_set_file(
    os.path.join(GENE_SETS_DIR, "go_biological_process")
)
gobp_dict = filter_gobp_dict(gobp_raw)
print(f"  GO:BP: {len(gobp_raw)} total -> {len(gobp_dict)} after filtering")

# ── Step 3: Run enrichment ───────────────────────────────────────────────────

print("\nStep 3: Running KEGG enrichment...")
kegg_enrichments = {}
kegg_enrichments_full = {}
for cell_type in CLUSTER_CELL_TYPE.values():
    full_df, top_df, lrt_df = run_enrichment(
        daily_cycler_acrophases[cell_type], kegg_dict, cell_type
    )
    kegg_enrichments_full[cell_type] = (full_df, lrt_df)
    kegg_enrichments[cell_type] = (top_df, lrt_df)

print("\nStep 4: Running GO:BP enrichment...")
gobp_enrichments = {}
gobp_enrichments_full = {}
for cell_type in CLUSTER_CELL_TYPE.values():
    full_df, top_df, lrt_df = run_enrichment(
        daily_cycler_acrophases[cell_type], gobp_dict, cell_type
    )
    gobp_enrichments_full[cell_type] = (full_df, lrt_df)
    gobp_enrichments[cell_type] = (top_df, lrt_df)

# ── Step 5: Export source data ───────────────────────────────────────────────

print("\nStep 5: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

export_enrichment_tables(SOURCE_DATA_DIR, "panel_a_kegg", kegg_enrichments_full, "pathway")
export_enrichment_tables(SOURCE_DATA_DIR, "panel_b_gobp", gobp_enrichments_full, "go_term")

for ct in ROSE_PLOT_CELL_TYPES:
    full_df, _ = gobp_enrichments_full[ct]
    if not full_df.empty:
        out = full_df.copy()
        out.index = [clean_gobp_name(n) for n in out.index]
        out.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_b_gobp_{ct}_readable.csv"))

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 6: Generate figure ──────────────────────────────────────────────────

print("\nStep 6: Generating figure...")

n_rose = len(ROSE_PLOT_CELL_TYPES)

fig = plt.figure(figsize=(8.5, 9), dpi=300)
fig.patch.set_facecolor("white")

gs = GridSpec(
    2, n_rose, figure=fig, hspace=0.55, wspace=0.65,
    height_ratios=[1.0, 1.0],
    left=0.08, right=0.92, top=0.95, bottom=0.04,
)


def make_rose_plot_clean(ax, enrich_df, lrt_df, title, clean_fn):
    """Wrapper that cleans pathway names before plotting."""
    if not enrich_df.empty:
        cleaned_df = enrich_df.copy()
        cleaned_df.index = [clean_fn(n) for n in cleaned_df.index]
        cleaned_lrt = lrt_df.copy()
        cleaned_lrt.index = [clean_fn(n) for n in cleaned_lrt.index]
    else:
        cleaned_df = enrich_df
        cleaned_lrt = lrt_df
    make_rose_plot(ax, cleaned_df, cleaned_lrt, title, is_tf=False)


# Row a: KEGG rose plots
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
    ax = fig.add_subplot(gs[0, i], projection="polar")
    enrich_df, lrt_df = kegg_enrichments[ct]
    make_rose_plot(ax, enrich_df, lrt_df, ct, is_tf=False)
    if i == 0:
        ax.text(-0.25, 1.22, "a", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.73, "KEGG\nPathways", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Row b: GO:BP rose plots
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
    ax = fig.add_subplot(gs[1, i], projection="polar")
    enrich_df, lrt_df = gobp_enrichments[ct]
    make_rose_plot_clean(ax, enrich_df, lrt_df, ct, clean_gobp_name)
    if i == 0:
        ax.text(-0.25, 1.22, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

fig.text(0.02, 0.27, "GO Biological\nProcess", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_2_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
