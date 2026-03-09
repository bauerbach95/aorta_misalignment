"""
Gene-set waveform enrichment analysis.

Supports two modes:
  Single-group: identify gene sets whose average waveform varies over the day.
  Two-group comparison: identify gene sets whose waveform DIFFERS between two
    conditions (e.g., aligned vs misaligned).

For single-group, we mesor-subtract each gene's waveform before averaging,
then test if the gene-set waveform varies across ZT (pairwise CI test).

For two-group, we compute per-gene difference waveforms (condition_a - condition_b),
average across the gene set, and test:
  (1) test_nonzero: does the difference exclude zero at any ZT?
  (2) test_time_varying: does the mesor-subtracted difference have temporal structure?
A gene set is "significant" if it passes test_nonzero.

Output:
  figures/group_waveform_enrichment/{slug}_{gs_type}.pdf
  figures/group_waveform_enrichment_source_data/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, CELL_TYPE_COLORS, FIGURES_DIR, ZT_LABELS,
    load_metrics, load_waveform_params, get_non_flat_genes,
    sample_waveforms, compute_gene_set_waveform, test_time_varying, test_nonzero,
    load_kegg_dict, load_tf_dict, format_label,
)

apply_style()

# ── Configuration ────────────────────────────────────────────────────────────

# Single-group analyses: list of condition strings
CONDITIONS_TO_RUN = [
    # "male aligned bmal1-control",
]

# Two-group comparisons: list of dicts
COMPARISONS_TO_RUN = [
    {
        "condition_a": "male aligned bmal1-control",
        "condition_b": "male misaligned bmal1-control",
        "label": "male aligned vs misaligned WT",
        "clusters": {"cluster_0": "SMC", "cluster_1" : "Fibroblast"},
    },
]

FLAT_LOG10_BF_THRESHOLD = 2.0
FRAC_CELL_DETECTED_MIN = 0.01
NUM_SAMPLES = 100
CREDIBLE_LEVEL = 0.95
MIN_GENES_IN_SET = 3
TOP_N_WAVEFORMS = 20
GENE_SET_TYPES = ["KEGG", "TF"]

FIGURES_SUBDIR = os.path.join(FIGURES_DIR, "group_waveform_enrichment")
SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "group_waveform_enrichment_source_data")

ZT_HOURS = [0, 6, 12, 18]


def condition_slug(cond):
    return cond.replace(" ", "_")


# ── Core enrichment routine ──────────────────────────────────────────────────

def run_enrichment_for_samples(all_samples, sampled_genes, gene_set_dicts, mode,
                               test_fn, credible_level=0.95):
    """Run gene-set enrichment on pre-computed samples.

    Args:
        all_samples: [num_genes, num_samples, 4]
        sampled_genes: list of gene names
        gene_set_dicts: {"KEGG": {...}, "TF": {...}}
        mode: "mesor_subtract" (single-group) or "average" (two-group diff)
        test_fn: function(gene_set_waveform, credible_level) -> (is_sig, details)
        credible_level: CI level for significance testing

    Returns:
        results_by_gs_type: {gs_type: list of significant set dicts}
    """
    gene_index = {g: i for i, g in enumerate(sampled_genes)}
    results_by_gs_type = {}

    for gs_type in GENE_SET_TYPES:
        gs_dict = gene_set_dicts[gs_type]
        significant_sets = []

        for gs_name, gs_genes in gs_dict.items():
            member_genes = [g for g in gs_genes if g in gene_index]
            if len(member_genes) < MIN_GENES_IN_SET:
                continue

            gs_waveform = compute_gene_set_waveform(
                all_samples, gene_index, member_genes, mode=mode
            )
            if gs_waveform is None:
                continue

            is_sig, details = test_fn(gs_waveform, credible_level)
            if not is_sig:
                continue

            mean_wave = gs_waveform.mean(axis=0)
            ci_lo = np.quantile(gs_waveform, 0.025, axis=0)
            ci_hi = np.quantile(gs_waveform, 0.975, axis=0)

            significant_sets.append({
                "name": gs_name,
                "num_genes": len(member_genes),
                "member_genes": member_genes,
                "mean_waveform": mean_wave,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "test_details": details,
                "max_amplitude": float(mean_wave.max() - mean_wave.min()),
            })

        significant_sets.sort(key=lambda x: -x["max_amplitude"])
        results_by_gs_type[gs_type] = significant_sets
        print(f"      {gs_type}: {len(significant_sets)} significant sets")

    return results_by_gs_type


# ── Source data export ───────────────────────────────────────────────────────

def export_source_data(results, slug, cell_type, is_comparison=False):
    """Export source data CSVs for one cell type's results."""
    for gs_type in GENE_SET_TYPES:
        sig_sets = results[gs_type]
        if not sig_sets:
            continue

        # Significant sets summary
        rows = []
        for s in sig_sets:
            row = {
                "gene_set": s["name"],
                "num_genes": s["num_genes"],
                "member_genes": ";".join(s["member_genes"]),
                "max_amplitude": round(s["max_amplitude"], 6),
            }
            for k, zt in enumerate(ZT_LABELS):
                row[f"mean_{zt}"] = round(float(s["mean_waveform"][k]), 6)
                row[f"ci_lo_{zt}"] = round(float(s["ci_lo"][k]), 6)
                row[f"ci_hi_{zt}"] = round(float(s["ci_hi"][k]), 6)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"{slug}_{cell_type}_{gs_type}_significant_sets.csv"),
            index=False,
        )

        # Test details
        detail_rows = []
        for s in sig_sets:
            for d in s["test_details"]:
                row = {"gene_set": s["name"]}
                row.update(d)
                detail_rows.append(row)
        pd.DataFrame(detail_rows).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"{slug}_{cell_type}_{gs_type}_test_details.csv"),
            index=False,
        )


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_waveform_panel(ax, top_sets, cell_type, gs_type, idx, ylabel, show_ci=True):
    """Plot waveform line plots with CI ribbons for one cell type."""
    if not top_sets:
        ax.text(0.5, 0.5, "No significant\nsets", transform=ax.transAxes,
                ha="center", va="center", fontsize=6, color="0.5", style="italic")
        ax.set_title(cell_type, fontsize=8, color=CELL_TYPE_COLORS[cell_type],
                     fontweight="semibold")
        ax.set_xticks(ZT_HOURS)
        ax.set_xticklabels(ZT_LABELS, fontsize=5.5)
        sns.despine(ax=ax)
        return

    n_sets = len(top_sets)
    if n_sets <= 8:
        colors = plt.cm.Set2(np.linspace(0, 1, 8))[:n_sets]
    elif n_sets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_sets]
    elif n_sets <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_sets]
    else:
        colors = plt.cm.gist_ncar(np.linspace(0, 0.9, n_sets))

    is_tf = gs_type == "TF"
    for i, gs in enumerate(top_sets):
        label = format_label(gs["name"], max_len=28, is_tf=is_tf)
        ax.plot(ZT_HOURS, gs["mean_waveform"], color=colors[i],
                linewidth=1.0, label=label, zorder=3)
        if show_ci:
            ax.fill_between(ZT_HOURS, gs["ci_lo"], gs["ci_hi"],
                            color=colors[i], alpha=0.15, zorder=2)

    ax.axhline(0, color="0.7", linewidth=0.4, linestyle="--", zorder=1)
    ax.set_title(cell_type, fontsize=8, color=CELL_TYPE_COLORS[cell_type],
                 fontweight="semibold")
    ax.set_xticks(ZT_HOURS)
    ax.set_xticklabels(ZT_LABELS, fontsize=5.5)
    ax.tick_params(axis="y", labelsize=5)
    if idx == 0:
        ax.set_ylabel(ylabel, fontsize=6.5)
    sns.despine(ax=ax)

    leg = ax.legend(
        fontsize=4, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.18), ncol=2 if n_sets > 5 else 1,
        handlelength=1.2, columnspacing=0.6,
    )
    for text in leg.get_texts():
        text.set_color("0.25")


def generate_figures(all_results, slug, title_prefix, cell_types, ylabel, show_ci=True):
    """Generate and save PDF figures for a set of cell types."""
    n_ct = len(cell_types)
    for gs_type in GENE_SET_TYPES:
        fig, axes = plt.subplots(1, n_ct, figsize=(3.2 * n_ct, 3.5), dpi=300,
                                 squeeze=False)
        axes = axes[0]
        fig.patch.set_facecolor("white")
        fig.suptitle(f"{gs_type} waveform enrichment \u2014 {title_prefix}",
                     fontsize=9, fontweight="semibold", y=1.0, color="0.15")

        for idx, ct in enumerate(cell_types):
            sig_sets = all_results[ct][gs_type]
            top_sets = sig_sets[:TOP_N_WAVEFORMS]
            plot_waveform_panel(axes[idx], top_sets, ct, gs_type, idx, ylabel, show_ci=show_ci)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        outpath = os.path.join(FIGURES_SUBDIR, f"{slug}_{gs_type}.pdf")
        fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        print(f"  Saved {outpath}")


# ── Parse arguments ───────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Gene-set waveform enrichment analysis")
parser.add_argument("--no-ci", action="store_true",
                    help="Omit confidence interval shading from waveform plots")
args = parser.parse_args()
SHOW_CI = not args.no_ci

# ── Step 1: Load gene set dictionaries ───────────────────────────────────────

print("Step 1: Loading gene set dictionaries...")
gene_set_dicts = {}
if "KEGG" in GENE_SET_TYPES:
    gene_set_dicts["KEGG"] = load_kegg_dict()
if "TF" in GENE_SET_TYPES:
    gene_set_dicts["TF"] = load_tf_dict()

os.makedirs(FIGURES_SUBDIR, exist_ok=True)
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

# ── Step 2: Single-group analyses ────────────────────────────────────────────

for condition in CONDITIONS_TO_RUN:
    print(f"\n=== Single-group: {condition} ===")
    slug = condition_slug(condition)
    all_results = {}

    for cluster, cell_type in CLUSTER_CELL_TYPE.items():
        print(f"  {cell_type}:")
        metrics_df = load_metrics(cluster, condition)
        alpha_df, beta_df, minmax_df = load_waveform_params(cluster, condition)
        non_flat = get_non_flat_genes(metrics_df, FLAT_LOG10_BF_THRESHOLD, FRAC_CELL_DETECTED_MIN)
        print(f"    {len(non_flat)} non-flat genes")

        if len(non_flat) < MIN_GENES_IN_SET:
            all_results[cell_type] = {gs: [] for gs in GENE_SET_TYPES}
            continue

        all_samples, sampled_genes = sample_waveforms(
            alpha_df, beta_df, minmax_df, sorted(non_flat), NUM_SAMPLES
        )
        all_results[cell_type] = run_enrichment_for_samples(
            all_samples, sampled_genes, gene_set_dicts,
            mode="mesor_subtract", test_fn=test_time_varying,
            credible_level=CREDIBLE_LEVEL,
        )
        export_source_data(all_results[cell_type], slug, cell_type)

    cell_types = list(CLUSTER_CELL_TYPE.values())
    generate_figures(all_results, slug, condition, cell_types,
                     ylabel="Mesor-subtracted\nlog rate", show_ci=SHOW_CI)

# ── Step 3: Two-group comparisons ────────────────────────────────────────────

for comp in COMPARISONS_TO_RUN:
    cond_a = comp["condition_a"]
    cond_b = comp["condition_b"]
    label = comp["label"]
    clusters = comp.get("clusters", CLUSTER_CELL_TYPE)
    slug = condition_slug(label)

    print(f"\n=== Comparison: {label} ===")
    print(f"    A: {cond_a}")
    print(f"    B: {cond_b}")
    all_results = {}

    for cluster, cell_type in clusters.items():
        print(f"  {cell_type}:")

        # Load metrics and params for both conditions
        metrics_a = load_metrics(cluster, cond_a)
        metrics_b = load_metrics(cluster, cond_b)
        alpha_a, beta_a, minmax_a = load_waveform_params(cluster, cond_a)
        alpha_b, beta_b, minmax_b = load_waveform_params(cluster, cond_b)

        # Union of non-flat genes from both conditions
        non_flat_a = get_non_flat_genes(metrics_a, FLAT_LOG10_BF_THRESHOLD, FRAC_CELL_DETECTED_MIN)
        non_flat_b = get_non_flat_genes(metrics_b, FLAT_LOG10_BF_THRESHOLD, FRAC_CELL_DETECTED_MIN)
        non_flat_union = sorted(non_flat_a | non_flat_b)
        print(f"    {len(non_flat_a)} non-flat (A), {len(non_flat_b)} non-flat (B), "
              f"{len(non_flat_union)} union")

        if len(non_flat_union) < MIN_GENES_IN_SET:
            all_results[cell_type] = {gs: [] for gs in GENE_SET_TYPES}
            continue

        # Sample waveforms from both conditions
        samples_a, genes_a = sample_waveforms(
            alpha_a, beta_a, minmax_a, non_flat_union, NUM_SAMPLES
        )
        samples_b, genes_b = sample_waveforms(
            alpha_b, beta_b, minmax_b, non_flat_union, NUM_SAMPLES
        )

        # Align to genes present in both
        genes_a_set, genes_b_set = set(genes_a), set(genes_b)
        shared_genes = sorted(genes_a_set & genes_b_set)
        print(f"    {len(shared_genes)} genes with params in both conditions")

        if len(shared_genes) < MIN_GENES_IN_SET:
            all_results[cell_type] = {gs: [] for gs in GENE_SET_TYPES}
            continue

        idx_a = [genes_a.index(g) for g in shared_genes]
        idx_b = [genes_b.index(g) for g in shared_genes]
        diff_samples = samples_a[idx_a] - samples_b[idx_b]  # [num_genes, num_samples, 4]

        all_results[cell_type] = run_enrichment_for_samples(
            diff_samples, shared_genes, gene_set_dicts,
            mode="average", test_fn=test_nonzero,
            credible_level=CREDIBLE_LEVEL,
        )
        export_source_data(all_results[cell_type], slug, cell_type, is_comparison=True)

    cell_types = list(clusters.values())
    generate_figures(all_results, slug, label, cell_types,
                     ylabel="Difference in\nlog rate (A \u2212 B)", show_ci=SHOW_CI)

print("\nDone.")
