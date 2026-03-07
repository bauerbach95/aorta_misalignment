"""
Gene-set waveform enrichment: identify KEGG pathways and TF target sets
whose average waveform varies significantly over the day within a group.

For each group (cell type x condition), we:
  1. Filter out flat genes (waveform not sufficiently better than flat model)
  2. Sample posterior waveforms for all non-flat genes
  3. For each gene set: compute mesor-subtracted average waveform, test if it
     varies across ZT0/6/12/18 (Bonferroni-corrected pairwise CIs)
  4. Plot significant gene sets as line plots with 95% CI ribbons

Output:
  figures/group_waveform_enrichment/{condition_slug}_{gs_type}.pdf
  figures/group_waveform_enrichment_source_data/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, CELL_TYPE_COLORS, FIGURES_DIR, ZT_LABELS,
    load_metrics, load_waveform_params, get_non_flat_genes,
    sample_waveforms, compute_gene_set_waveform, test_time_varying,
    load_kegg_dict, load_tf_dict, format_label, clean_tf_name,
)

apply_style()

# ── Configuration ────────────────────────────────────────────────────────────

CONDITIONS_TO_RUN = [
    "male aligned bmal1-control",
]

FLAT_LOG10_BF_THRESHOLD = 2.0
FRAC_CELL_DETECTED_MIN = 0.01
NUM_SAMPLES = 100
CREDIBLE_LEVEL = 0.95
MIN_GENES_IN_SET = 5
TOP_N_WAVEFORMS = 10
GENE_SET_TYPES = ["KEGG", "TF"]

FIGURES_SUBDIR = os.path.join(FIGURES_DIR, "group_waveform_enrichment")
SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "group_waveform_enrichment_source_data")

ZT_HOURS = [0, 6, 12, 18]


def condition_slug(cond):
    return cond.replace(" ", "_")


# ── Step 1: Load gene set dictionaries ───────────────────────────────────────

print("Step 1: Loading gene set dictionaries...")
gene_set_dicts = {}
if "KEGG" in GENE_SET_TYPES:
    gene_set_dicts["KEGG"] = load_kegg_dict()
if "TF" in GENE_SET_TYPES:
    gene_set_dicts["TF"] = load_tf_dict()

# ── Step 2: Main analysis loop ───────────────────────────────────────────────

print("\nStep 2: Running waveform enrichment analysis...")

# results[condition][cell_type][gs_type] = list of significant set dicts
results = {}

for condition in CONDITIONS_TO_RUN:
    print(f"\n  Condition: {condition}")
    results[condition] = {}

    for cluster, cell_type in CLUSTER_CELL_TYPE.items():
        print(f"    {cell_type}:")
        results[condition][cell_type] = {}

        # Load data
        metrics_df = load_metrics(cluster, condition)
        alpha_df, beta_df, minmax_df = load_waveform_params(cluster, condition)

        # Filter non-flat genes
        non_flat = get_non_flat_genes(metrics_df, FLAT_LOG10_BF_THRESHOLD, FRAC_CELL_DETECTED_MIN)
        print(f"      {len(non_flat)} non-flat genes")

        if len(non_flat) < MIN_GENES_IN_SET:
            for gs_type in GENE_SET_TYPES:
                results[condition][cell_type][gs_type] = []
            continue

        # Sample all non-flat genes once
        all_samples, sampled_genes = sample_waveforms(
            alpha_df, beta_df, minmax_df, sorted(non_flat), NUM_SAMPLES
        )
        gene_index = {g: i for i, g in enumerate(sampled_genes)}

        # Run enrichment for each gene set type
        for gs_type in GENE_SET_TYPES:
            gs_dict = gene_set_dicts[gs_type]
            significant_sets = []

            for gs_name, gs_genes in gs_dict.items():
                member_genes = [g for g in gs_genes if g in gene_index]
                if len(member_genes) < MIN_GENES_IN_SET:
                    continue

                gs_waveform = compute_gene_set_waveform(
                    all_samples, gene_index, member_genes, mode="mesor_subtract"
                )
                if gs_waveform is None:
                    continue

                is_sig, pairwise = test_time_varying(gs_waveform, CREDIBLE_LEVEL)
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
                    "pairwise_results": pairwise,
                    "max_amplitude": float(mean_wave.max() - mean_wave.min()),
                })

            significant_sets.sort(key=lambda x: -x["max_amplitude"])
            results[condition][cell_type][gs_type] = significant_sets
            print(f"      {gs_type}: {len(significant_sets)} significant sets")

# ── Step 3: Export source data ───────────────────────────────────────────────

print("\nStep 3: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

for condition in CONDITIONS_TO_RUN:
    slug = condition_slug(condition)
    for cell_type in CLUSTER_CELL_TYPE.values():
        for gs_type in GENE_SET_TYPES:
            sig_sets = results[condition][cell_type][gs_type]
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

            # Pairwise test details
            pw_rows = []
            for s in sig_sets:
                for pr in s["pairwise_results"]:
                    pw_rows.append({
                        "gene_set": s["name"],
                        "pair": pr["pair"],
                        "mean_diff": round(pr["mean_diff"], 6),
                        "ci_lo": round(pr["ci_lo"], 6),
                        "ci_hi": round(pr["ci_hi"], 6),
                        "excludes_zero": pr["excludes_zero"],
                    })
            pd.DataFrame(pw_rows).to_csv(
                os.path.join(SOURCE_DATA_DIR, f"{slug}_{cell_type}_{gs_type}_pairwise_tests.csv"),
                index=False,
            )

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 4: Generate figures ─────────────────────────────────────────────────

print("\nStep 4: Generating figures...")
os.makedirs(FIGURES_SUBDIR, exist_ok=True)

all_cell_types = list(CLUSTER_CELL_TYPE.values())
n_ct = len(all_cell_types)

for condition in CONDITIONS_TO_RUN:
    slug = condition_slug(condition)
    for gs_type in GENE_SET_TYPES:
        fig, axes = plt.subplots(1, n_ct, figsize=(3.2 * n_ct, 3.5), dpi=300)
        fig.patch.set_facecolor("white")
        fig.suptitle(f"{gs_type} waveform enrichment — {condition}",
                     fontsize=9, fontweight="semibold", y=1.0, color="0.15")

        for idx, ct in enumerate(all_cell_types):
            ax = axes[idx]
            sig_sets = results[condition][ct][gs_type]
            top_sets = sig_sets[:TOP_N_WAVEFORMS]

            if not top_sets:
                ax.text(0.5, 0.5, "No significant\nsets", transform=ax.transAxes,
                        ha="center", va="center", fontsize=6, color="0.5", style="italic")
                ax.set_title(ct, fontsize=8, color=CELL_TYPE_COLORS[ct], fontweight="semibold")
                ax.set_xticks(ZT_HOURS)
                ax.set_xticklabels(ZT_LABELS, fontsize=5.5)
                sns.despine(ax=ax)
                continue

            n_sets = len(top_sets)
            if n_sets <= 8:
                colors = plt.cm.Set2(np.linspace(0, 1, 8))[:n_sets]
            else:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_sets]

            for i, gs in enumerate(top_sets):
                is_tf = gs_type == "TF"
                label = format_label(gs["name"], max_len=28, is_tf=is_tf)
                ax.plot(ZT_HOURS, gs["mean_waveform"], color=colors[i],
                        linewidth=1.0, label=label, zorder=3)
                ax.fill_between(ZT_HOURS, gs["ci_lo"], gs["ci_hi"],
                                color=colors[i], alpha=0.15, zorder=2)

            ax.axhline(0, color="0.7", linewidth=0.4, linestyle="--", zorder=1)
            ax.set_title(ct, fontsize=8, color=CELL_TYPE_COLORS[ct], fontweight="semibold")
            ax.set_xticks(ZT_HOURS)
            ax.set_xticklabels(ZT_LABELS, fontsize=5.5)
            ax.tick_params(axis="y", labelsize=5)
            if idx == 0:
                ax.set_ylabel("Mesor-subtracted\nlog rate", fontsize=6.5)
            sns.despine(ax=ax)

            leg = ax.legend(
                fontsize=4, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.18), ncol=2 if n_sets > 5 else 1,
                handlelength=1.2, columnspacing=0.6,
            )
            for text in leg.get_texts():
                text.set_color("0.25")

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        outpath = os.path.join(FIGURES_SUBDIR, f"{slug}_{gs_type}.pdf")
        fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
        print(f"  Saved {outpath}")

print("\nDone.")
