"""
Generate Figure 3: Enhanced rhythmicity of female cell types.

Subplots:
  a) Histograms: fraction of amplitude samples larger in female (4 cell types)
  b) Rhythmicity sweep: # cyclers vs BF threshold, downsampled (SMC, Fibroblast)
  c) Two-group waveform enrichment: female vs male (KEGG/TF, SMC/Fibroblast)

Output:
  figures/fig_3.pdf
  figures/fig_3_source_data/
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
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, CELL_TYPE_COLORS,
    FIGURES_DIR, WAVEFORM_BF_MIN, ZT_LABELS,
    load_metrics, filter_cyclers, load_waveform_params, sample_waveforms,
    compute_amplitude,
    load_downsampled_metrics, filter_cyclers_downsampled,
    get_non_flat_genes, compute_gene_set_waveform, test_nonzero,
    load_kegg_dict, load_tf_dict, format_label,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_3_source_data")
NUM_SAMPLES = 5000
ZT_HOURS = [0, 6, 12, 18]

# ── Step 1: Panel a — amplitude comparison ──────────────────────────────────

print("Step 1: Amplitude comparison (female vs male)...")

amp_fracs = {}  # {cell_type: array of frac_female_larger}

for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    male_df = load_metrics(cluster, ALIGNED_CONDITIONS["male"])
    female_df = load_metrics(cluster, ALIGNED_CONDITIONS["female"])

    # Genes cycling in either sex
    male_cyclers = filter_cyclers(male_df)
    female_cyclers = filter_cyclers(female_df)
    union_cyclers = sorted(male_cyclers | female_cyclers)

    if len(union_cyclers) < 5:
        print(f"  {cell_type}: SKIP (only {len(union_cyclers)} union cyclers)")
        amp_fracs[cell_type] = np.array([])
        continue

    # Sample waveforms for both sexes
    alpha_m, beta_m, mm_m = load_waveform_params(cluster, ALIGNED_CONDITIONS["male"])
    alpha_f, beta_f, mm_f = load_waveform_params(cluster, ALIGNED_CONDITIONS["female"])

    samp_m, genes_m = sample_waveforms(alpha_m, beta_m, mm_m, union_cyclers, NUM_SAMPLES)
    samp_f, genes_f = sample_waveforms(alpha_f, beta_f, mm_f, union_cyclers, NUM_SAMPLES)

    # Align genes present in both
    idx_m = {g: i for i, g in enumerate(genes_m)}
    idx_f = {g: i for i, g in enumerate(genes_f)}
    common = [g for g in union_cyclers if g in idx_m and g in idx_f]

    if not common:
        amp_fracs[cell_type] = np.array([])
        continue

    ord_m = [idx_m[g] for g in common]
    ord_f = [idx_f[g] for g in common]

    amp_m = compute_amplitude(samp_m[ord_m])  # [n_genes, N]
    amp_f = compute_amplitude(samp_f[ord_f])

    frac_f_larger = (amp_f > amp_m).mean(axis=1)
    amp_fracs[cell_type] = frac_f_larger
    median_frac = np.median(frac_f_larger)
    print(f"  {cell_type}: {len(common)} genes, median frac = {median_frac:.3f}")

# ── Step 2: Panel b — rhythmicity sweep (downsampled) ──────────────────────

print("\nStep 2: Rhythmicity sweep (downsampled data)...")

SWEEP_CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}
bf_thresholds = np.concatenate([np.arange(0.5, 3.0, 0.25), np.arange(3, 11, 1.0)])
sweep_data = {}  # {cell_type: {thresh: {male, female, shared, unique_m, unique_f}}}

for cluster, cell_type in SWEEP_CLUSTERS.items():
    male_df = load_downsampled_metrics(cluster, "male aligned bmal1-control")
    female_df = load_downsampled_metrics(cluster, "female aligned bmal1-control")

    results = []
    for thresh in bf_thresholds:
        m_cyc = filter_cyclers_downsampled(male_df, bf_threshold=thresh)
        f_cyc = filter_cyclers_downsampled(female_df, bf_threshold=thresh)
        shared = m_cyc & f_cyc
        results.append({
            "bf_threshold": thresh,
            "male": len(m_cyc),
            "female": len(f_cyc),
            "shared": len(shared),
            "unique_male": len(m_cyc - f_cyc),
            "unique_female": len(f_cyc - m_cyc),
        })
    sweep_data[cell_type] = pd.DataFrame(results)
    print(f"  {cell_type}: male={results[0]['male']}, female={results[0]['female']} at BF≥{bf_thresholds[0]}")

# ── Step 3: Panel c — two-group waveform enrichment (female vs male) ───────

print("\nStep 3: Two-group waveform enrichment (female vs male)...")

ENRICH_CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}
NUM_ENRICH_SAMPLES = 100
MIN_GENES_IN_SET = 3
CREDIBLE_LEVEL = 0.95
TOP_N_WAVEFORMS = 15
GS_TYPES = ["KEGG", "TF"]

kegg_dict = load_kegg_dict()
tf_dict = load_tf_dict()
gene_set_dicts = {"KEGG": kegg_dict, "TF": tf_dict}

enrich_results = {}  # {cell_type: {gs_type: [list of sig set dicts]}}

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

    # Align to shared genes
    set_m, set_f = set(genes_m), set(genes_f)
    shared_genes = sorted(set_m & set_f)
    idx_m_map = {g: i for i, g in enumerate(genes_m)}
    idx_f_map = {g: i for i, g in enumerate(genes_f)}
    ord_m = [idx_m_map[g] for g in shared_genes]
    ord_f = [idx_f_map[g] for g in shared_genes]

    diff_samples = samp_f[ord_f] - samp_m[ord_m]  # female - male
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

# ── Step 4: Export source data ──────────────────────────────────────────────

print("\nStep 4: Exporting source data...")
os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

# Panel a
for ct, fracs in amp_fracs.items():
    if len(fracs) > 0:
        pd.DataFrame({"frac_female_amplitude_larger": fracs}).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_a_amplitude_fracs_{ct}.csv"), index=False)

# Panel b
for ct, df in sweep_data.items():
    df.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_b_sweep_{ct}.csv"), index=False)

# Panel c
for ct, ct_res in enrich_results.items():
    for gs_type, sig_sets in ct_res.items():
        if not sig_sets:
            continue
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
            os.path.join(SOURCE_DATA_DIR, f"panel_c_{gs_type}_{ct}.csv"), index=False)

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 5: Generate figure ─────────────────────────────────────────────────

print("\nStep 5: Generating figure...")

fig = plt.figure(figsize=(8.5, 12), dpi=300)
fig.patch.set_facecolor("white")

gs_main = GridSpec(3, 1, figure=fig, hspace=0.5,
                   height_ratios=[0.55, 0.55, 1.0],
                   left=0.08, right=0.95, top=0.96, bottom=0.05)

# ── Row a: amplitude histograms ─────────────────────────────────────────────

gs_a = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[0], wspace=0.35)
all_cell_types = ["SMC", "Fibroblast", "EC", "Macrophage"]

for i, ct in enumerate(all_cell_types):
    ax = fig.add_subplot(gs_a[0, i])
    fracs = amp_fracs[ct]
    if len(fracs) > 0:
        ax.hist(fracs, bins=30, color=CELL_TYPE_COLORS[ct], edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.axvline(0.5, color="k", linestyle="--", linewidth=0.6, alpha=0.5)
        median_val = np.median(fracs)
        ax.set_title(ct, fontsize=8, fontweight="semibold",
                     color=CELL_TYPE_COLORS[ct])
        ax.text(0.97, 0.95, f"n={len(fracs)}\nmed={median_val:.2f}",
                transform=ax.transAxes, fontsize=5, ha="right", va="top", color="0.35")
    else:
        ax.text(0.5, 0.5, "Too few\ngenes", transform=ax.transAxes,
                ha="center", va="center", fontsize=6, color="0.5", style="italic")
        ax.set_title(ct, fontsize=8, fontweight="semibold",
                     color=CELL_TYPE_COLORS[ct])
    ax.set_xlabel("Fraction amplitude\nlarger in female" if i == 1 else "", fontsize=6)
    ax.set_ylabel("Number of genes" if i == 0 else "", fontsize=6.5)
    ax.set_xlim(0, 1)
    sns.despine(ax=ax)
    if i == 0:
        ax.text(-0.2, 1.12, "a", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# ── Row b: rhythmicity sweep ────────────────────────────────────────────────

gs_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], wspace=0.35)
sweep_colors = {
    "male": "#5A7EBD", "female": "#E8963E",
    "unique_male": "#4CAF50", "unique_female": "#E74C3C",
    "shared": "#9B59B6",
}

for i, ct in enumerate(SWEEP_CLUSTERS.values()):
    ax = fig.add_subplot(gs_b[0, i])
    df = sweep_data[ct]
    for key, color in sweep_colors.items():
        ax.plot(df["bf_threshold"], df[key], color=color,
                linewidth=1.2, label=key.replace("_", " "))
    ax.set_xlabel("Log10 circadian bayes factor threshold", fontsize=6.5)
    ax.set_ylabel("Number of circadian cyclers" if i == 0 else "", fontsize=6.5)
    ax.set_title(ct, fontsize=8, fontweight="semibold",
                 color=CELL_TYPE_COLORS[ct])
    ax.legend(fontsize=5, frameon=False, loc="upper right")
    ax.set_xlim(bf_thresholds[0], bf_thresholds[-1])
    sns.despine(ax=ax)
    if i == 0:
        ax.text(-0.17, 1.12, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# ── Row c: waveform enrichment line plots ────────────────────────────────────

gs_c = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[2], hspace=0.55, wspace=0.35)
enrich_cell_types = list(ENRICH_CLUSTERS.values())

for row_idx, gs_type in enumerate(GS_TYPES):
    for col_idx, ct in enumerate(enrich_cell_types):
        ax = fig.add_subplot(gs_c[row_idx, col_idx])
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

            is_tf = gs_type == "TF"
            for j, gs in enumerate(top_sets):
                label = format_label(gs["name"], max_len=28, is_tf=is_tf)
                ax.plot(ZT_HOURS, gs["mean_waveform"], color=colors[j],
                        linewidth=1.0, label=label, zorder=3)
                ax.fill_between(ZT_HOURS, gs["ci_lo"], gs["ci_hi"],
                                color=colors[j], alpha=0.15, zorder=2)
            ax.axhline(0, color="0.7", linewidth=0.4, linestyle="--", zorder=1)
            leg = ax.legend(
                fontsize=3.5, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.2), ncol=2 if n_sets > 5 else 1,
                handlelength=1.2, columnspacing=0.6)
            for text in leg.get_texts():
                text.set_color("0.25")

        title = ct
        if row_idx == 0:
            title = f"{ct}"
        ax.set_title(title, fontsize=7, fontweight="semibold",
                     color=CELL_TYPE_COLORS[ct])
        ax.set_xticks(ZT_HOURS)
        ax.set_xticklabels(ZT_LABELS, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5)
        if col_idx == 0:
            ax.set_ylabel(f"{gs_type}\nDiff log rate (F\u2212M)", fontsize=6)
        sns.despine(ax=ax)

        if row_idx == 0 and col_idx == 0:
            ax.text(-0.22, 1.15, "c", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_3.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
