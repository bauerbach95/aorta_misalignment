"""
Generate Supplementary Figure 6: Gene-set waveform enrichment (Misaligned vs Aligned, male).

Two-group waveform enrichment analysis showing which pathways/TFs are
differentially expressed at each ZT timepoint under misalignment,
plus additional proteostasis gene waveforms.

Subplots:
  a) KEGG pathway waveform enrichment (SMC / Fibroblast)
  b) TF waveform enrichment (SMC / Fibroblast)
  c) Additional proteostasis gene waveforms

Output:
  figures/fig_6_supp.pdf
  figures/fig_6_supp_source_data/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, MISALIGNED_CONDITIONS,
    CELL_TYPE_COLORS, FIGURES_DIR, ZT_LABELS,
    load_metrics, load_waveform_params, sample_waveforms,
    get_non_flat_genes, compute_gene_set_waveform, test_nonzero,
    load_kegg_dict, load_tf_dict, format_label,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_6_supp_source_data")
ZT_HOURS = [0, 6, 12, 18]
SEX = "male"

ENRICH_CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}
NUM_ENRICH_SAMPLES = 100
MIN_GENES_IN_SET = 3
CREDIBLE_LEVEL = 0.95
TOP_N_WAVEFORMS = 15

COND_AL = ALIGNED_CONDITIONS[SEX]
COND_MIS = MISALIGNED_CONDITIONS[SEX]


def clean_kegg_name(name):
    return name


def clean_tf_name(name):
    return name


# ── Step 1: Load gene set databases ──────────────────────────────────────────

print("Step 1: Loading gene set databases...")

kegg_dict = load_kegg_dict()
tf_dict = load_tf_dict()
print(f"  KEGG: {len(kegg_dict)} sets")
print(f"  TF: {len(tf_dict)} sets")

GS_TYPES = ["KEGG", "TF"]
gene_set_dicts = {
    "KEGG": kegg_dict,
    "TF": tf_dict,
}

# ── Step 2: Two-group waveform enrichment (misaligned vs aligned, male) ──────

print("\nStep 2: Two-group waveform enrichment (misaligned vs aligned, male)...")

enrich_results = {}

for cluster, cell_type in ENRICH_CLUSTERS.items():
    print(f"  {cell_type}:")

    metrics_al = load_metrics(cluster, COND_AL)
    metrics_mis = load_metrics(cluster, COND_MIS)
    alpha_al, beta_al, mm_al = load_waveform_params(cluster, COND_AL)
    alpha_mis, beta_mis, mm_mis = load_waveform_params(cluster, COND_MIS)

    non_flat_al = get_non_flat_genes(metrics_al)
    non_flat_mis = get_non_flat_genes(metrics_mis)
    non_flat_union = sorted(non_flat_al | non_flat_mis)
    print(f"    {len(non_flat_union)} non-flat genes (union)")

    samp_al, genes_al = sample_waveforms(alpha_al, beta_al, mm_al, non_flat_union, NUM_ENRICH_SAMPLES)
    samp_mis, genes_mis = sample_waveforms(alpha_mis, beta_mis, mm_mis, non_flat_union, NUM_ENRICH_SAMPLES)

    set_al, set_mis = set(genes_al), set(genes_mis)
    shared_genes = sorted(set_al & set_mis)
    idx_al_map = {g: i for i, g in enumerate(genes_al)}
    idx_mis_map = {g: i for i, g in enumerate(genes_mis)}
    ord_al = [idx_al_map[g] for g in shared_genes]
    ord_mis = [idx_mis_map[g] for g in shared_genes]

    diff_samples = samp_mis[ord_mis] - samp_al[ord_al]  # misaligned - aligned
    gene_index = {g: i for i, g in enumerate(shared_genes)}

    ct_results = {}
    for gs_type in GS_TYPES:
        gs_dict = gene_set_dicts[gs_type]
        sig_sets = []
        for gs_name, gs_genes in gs_dict.items():
            members = [g for g in gs_genes if g in gene_index]
            if len(members) < MIN_GENES_IN_SET:
                continue
            gs_waveform = compute_gene_set_waveform(diff_samples, gene_index, members, mode="average")
            if gs_waveform is None:
                continue
            is_sig, details = test_nonzero(gs_waveform, CREDIBLE_LEVEL)
            if not is_sig:
                continue
            mean_wave = gs_waveform.mean(axis=0)
            sig_sets.append({
                "name": gs_name,
                "num_genes": len(members),
                "member_genes": members,
                "mean_waveform": mean_wave,
                "ci_lo": np.quantile(gs_waveform, 0.025, axis=0),
                "ci_hi": np.quantile(gs_waveform, 0.975, axis=0),
                "max_amplitude": float(mean_wave.max() - mean_wave.min()),
            })
        sig_sets.sort(key=lambda x: -x["max_amplitude"])
        ct_results[gs_type] = sig_sets
        print(f"    {gs_type}: {len(sig_sets)} significant sets")
    enrich_results[cell_type] = ct_results

# ── Step 3: Export source data ───────────────────────────────────────────────

print("\nStep 3: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

for ct, ct_res in enrich_results.items():
    for gs_type, sig_sets in ct_res.items():
        if not sig_sets:
            continue
        gs_label = gs_type.replace(":", "").lower()
        rows = []
        for s in sig_sets:
            row = {"gene_set": s["name"], "num_genes": s["num_genes"],
                   "member_genes": ";".join(s["member_genes"]),
                   "max_amplitude": round(s["max_amplitude"], 6)}
            for k, zt in enumerate(ZT_LABELS):
                row[f"mean_{zt}"] = round(float(s["mean_waveform"][k]), 6)
                row[f"ci_lo_{zt}"] = round(float(s["ci_lo"][k]), 6)
                row[f"ci_hi_{zt}"] = round(float(s["ci_hi"][k]), 6)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_{gs_label}_{ct}.csv"), index=False)

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 4: Load supplementary proteostasis gene waveforms ──────────────────

print("\nStep 4: Loading supplementary proteostasis gene waveforms...")

from circadian_utils import plot_posterior_violins

PROTEO_SUPP_GENES = ["Hspa1b", "Dnajb1", "Dnajb4", "Dnaja1", "Hsph1", "Ankrd1", "Paip2b"]
COND_COLORS = {"Aligned": "#4878CF", "Misaligned": "#D65F5F"}

alpha_al_smc, beta_al_smc, mm_al_smc = load_waveform_params("cluster_0", COND_AL)
alpha_mis_smc, beta_mis_smc, mm_mis_smc = load_waveform_params("cluster_0", COND_MIS)

proteo_supp_waveforms = {}
for gene in PROTEO_SUPP_GENES:
    proteo_supp_waveforms[gene] = {}
    for cond_label, (a, b, m) in [("Aligned", (alpha_al_smc, beta_al_smc, mm_al_smc)),
                                   ("Misaligned", (alpha_mis_smc, beta_mis_smc, mm_mis_smc))]:
        samp, genes = sample_waveforms(a, b, m, [gene], 5000)
        if genes:
            proteo_supp_waveforms[gene][cond_label] = samp[0]
found_supp = [g for g in PROTEO_SUPP_GENES if proteo_supp_waveforms[g]]
print(f"  Found: {', '.join(found_supp)}")

# ── Step 5: Generate figure ──────────────────────────────────────────────────

print("\nStep 5: Generating figure...")

n_proteo_cols = min(4, len(found_supp))
n_proteo_rows = (len(found_supp) + n_proteo_cols - 1) // n_proteo_cols
proteo_height = 0.3 * n_proteo_rows

from matplotlib.gridspec import GridSpecFromSubplotSpec

fig = plt.figure(figsize=(11, 6 + proteo_height * 4), dpi=300)
fig.patch.set_facecolor("white")

gs_top = GridSpec(3, 1, figure=fig, hspace=0.45,
                  height_ratios=[1, 1, proteo_height * n_proteo_rows],
                  left=0.08, right=0.55, top=0.97, bottom=0.04)

gs_enrich = [GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_top[i], wspace=0.85) for i in range(2)]

enrich_cell_types = list(ENRICH_CLUSTERS.values())

clean_fns = {
    "KEGG": clean_kegg_name,
    "TF": clean_tf_name,
}

panel_labels = ["a", "b"]

for row_idx, gs_type in enumerate(GS_TYPES):
    for col_idx, ct in enumerate(enrich_cell_types):
        ax = fig.add_subplot(gs_enrich[row_idx][0, col_idx])
        sig_sets = enrich_results[ct][gs_type]
        top_sets = sig_sets[:TOP_N_WAVEFORMS]

        if not top_sets:
            ax.text(0.5, 0.5, "No significant\nsets", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="0.5", style="italic")
        else:
            n_sets = len(top_sets)
            if n_sets <= 8:
                colors = plt.cm.Set2(np.linspace(0, 1, 8))[:n_sets]
            elif n_sets <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_sets]
            else:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_sets]

            clean_fn = clean_fns[gs_type]
            for j, gs in enumerate(top_sets):
                label = format_label(clean_fn(gs["name"]), max_len=32)
                ax.plot(ZT_HOURS, gs["mean_waveform"], color=colors[j],
                        linewidth=1.0, label=label, zorder=3)
            ax.axhline(0, color="0.7", linewidth=0.4, linestyle="--", zorder=1)
            leg = ax.legend(
                fontsize=3.5, frameon=False, loc="upper left",
                bbox_to_anchor=(1.08, 1.0), ncol=1,
                handlelength=1.2, columnspacing=0.6)
            for text in leg.get_texts():
                text.set_color("0.25")

        ax.set_title(ct, fontsize=7, fontweight="semibold",
                     color=CELL_TYPE_COLORS[ct])
        ax.set_xticks(ZT_HOURS)
        ax.set_xticklabels(ZT_LABELS, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5)
        if col_idx == 0:
            ax.set_ylabel(f"{gs_type}\nDiff log rate (Mis\u2212Al)", fontsize=6)
        sns.despine(ax=ax)

        if col_idx == 0:
            ax.text(-0.12, 1.15, panel_labels[row_idx], transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")

# ── Row e: supplementary proteostasis gene waveforms ──────────────────────

gs_proteo = GridSpecFromSubplotSpec(n_proteo_rows, n_proteo_cols,
                                    subplot_spec=gs_top[2],
                                    hspace=0.5, wspace=0.35)

for idx, gene in enumerate(found_supp):
    r = idx // n_proteo_cols
    c = idx % n_proteo_cols
    ax = fig.add_subplot(gs_proteo[r, c])
    gene_samps = {cl: s for cl, s in proteo_supp_waveforms[gene].items()}
    if gene_samps:
        plot_posterior_violins(ax, gene_samps, COND_COLORS, width=2.0, alpha=0.5)
    ax.set_title(gene, fontsize=6.5, fontweight="semibold", color="0.2")
    ax.tick_params(axis="y", labelsize=4.5)
    if c == 0:
        ax.set_ylabel("Log10 rate", fontsize=5.5)
    if idx < 2:
        ax.legend(fontsize=4, frameon=False)
    sns.despine(ax=ax)
    if idx == 0:
        ax.text(-0.3, 1.2, "c", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_6_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
