"""
Generate Supplementary Figure 4: Reactome and GO:BP waveform enrichment (F vs M).

Extends Figure 3c (KEGG/TF) with additional gene set databases for the
two-group waveform enrichment analysis (female - male) in SMC and Fibroblast.

Subplots:
  a) Reactome pathway waveform enrichment (SMC, Fibroblast)
  b) GO Biological Process waveform enrichment (SMC, Fibroblast)

Output:
  figures/fig_3_supp.pdf
  figures/fig_3_supp_source_data/
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, CELL_TYPE_COLORS,
    FIGURES_DIR, ZT_LABELS, GENE_SETS_DIR,
    load_metrics, load_waveform_params, sample_waveforms,
    get_non_flat_genes, compute_gene_set_waveform, test_nonzero,
    load_reactome_dict, clean_reactome_name, parse_gsea_set_file, format_label,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_3_supp_source_data")
ZT_HOURS = [0, 6, 12, 18]

ENRICH_CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}
NUM_ENRICH_SAMPLES = 100
MIN_GENES_IN_SET = 3
CREDIBLE_LEVEL = 0.95
TOP_N_WAVEFORMS = 15

# GO:BP filtering (same as gen_fig_2_supp.py)
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


# ── Step 1: Load gene set databases ──────────────────────────────────────────

print("Step 1: Loading gene set databases...")

reactome_dict = load_reactome_dict()

gobp_raw = parse_gsea_set_file(
    os.path.join(GENE_SETS_DIR, "go_biological_process")
)
gobp_dict = filter_gobp_dict(gobp_raw)
print(f"  GO:BP: {len(gobp_raw)} total -> {len(gobp_dict)} after filtering")

GS_TYPES = ["Reactome", "GO:BP"]
gene_set_dicts = {"Reactome": reactome_dict, "GO:BP": gobp_dict}

# ── Step 2: Run two-group waveform enrichment (female vs male) ───────────────

print("\nStep 2: Two-group waveform enrichment (female vs male)...")

enrich_results = {}

for cluster, cell_type in ENRICH_CLUSTERS.items():
    print(f"  {cell_type}:")

    metrics_m = load_metrics(cluster, ALIGNED_CONDITIONS["male"])
    metrics_f = load_metrics(cluster, ALIGNED_CONDITIONS["female"])
    alpha_m, beta_m, mm_m = load_waveform_params(cluster, ALIGNED_CONDITIONS["male"])
    alpha_f, beta_f, mm_f = load_waveform_params(cluster, ALIGNED_CONDITIONS["female"])

    non_flat_m = get_non_flat_genes(metrics_m)
    non_flat_f = get_non_flat_genes(metrics_f)
    non_flat_union = sorted(non_flat_m | non_flat_f)
    print(f"    {len(non_flat_union)} non-flat genes (union)")

    samp_m, genes_m = sample_waveforms(alpha_m, beta_m, mm_m, non_flat_union, NUM_ENRICH_SAMPLES)
    samp_f, genes_f = sample_waveforms(alpha_f, beta_f, mm_f, non_flat_union, NUM_ENRICH_SAMPLES)

    set_m, set_f = set(genes_m), set(genes_f)
    shared_genes = sorted(set_m & set_f)
    idx_m_map = {g: i for i, g in enumerate(genes_m)}
    idx_f_map = {g: i for i, g in enumerate(genes_f)}
    ord_m = [idx_m_map[g] for g in shared_genes]
    ord_f = [idx_f_map[g] for g in shared_genes]

    diff_samples = samp_f[ord_f] - samp_m[ord_m]
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

# ── Step 4: Generate figure ──────────────────────────────────────────────────

print("\nStep 4: Generating figure...")

fig = plt.figure(figsize=(8.5, 9), dpi=300)
fig.patch.set_facecolor("white")

gs_main = GridSpec(2, 2, figure=fig, hspace=0.85, wspace=0.35,
                   left=0.10, right=0.95, top=0.95, bottom=0.08)

enrich_cell_types = list(ENRICH_CLUSTERS.values())

# Name cleaning functions per gene set type
clean_fns = {
    "Reactome": clean_reactome_name,
    "GO:BP": clean_gobp_name,
}

for row_idx, gs_type in enumerate(GS_TYPES):
    for col_idx, ct in enumerate(enrich_cell_types):
        ax = fig.add_subplot(gs_main[row_idx, col_idx])
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
                fontsize=3.5, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.2), ncol=2 if n_sets > 5 else 1,
                handlelength=1.2, columnspacing=0.6)
            for text in leg.get_texts():
                text.set_color("0.25")

        ax.set_title(ct, fontsize=7, fontweight="semibold",
                     color=CELL_TYPE_COLORS[ct])
        ax.set_xticks(ZT_HOURS)
        ax.set_xticklabels(ZT_LABELS, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5)
        if col_idx == 0:
            ax.set_ylabel(f"{gs_type}\nDiff log rate (F\u2212M)", fontsize=6)
        sns.despine(ax=ax)

        if row_idx == 0 and col_idx == 0:
            ax.text(-0.22, 1.15, "a", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")
        if row_idx == 1 and col_idx == 0:
            ax.text(-0.22, 1.15, "b", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_3_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
