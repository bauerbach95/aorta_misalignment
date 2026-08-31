"""
Generate Supplementary Figure 7: Acute misalignment effects in females.

Female counterpart to Figure 4 (male primary).

Subplots:
  a) Amplitude loss histograms -- aligned vs misaligned (4 cell types, female)
  b) Acrophase shift scatter -- aligned vs misaligned (4 cell types, female)
  c) Clock relative timing in SMCs -- Dbp/Nr1d1 vs Arntl phase shifts (female)
  d) CCR2/CX3CR1 waveforms in macrophages -- aligned vs misaligned (female)
  e) KEGG misalignment enrichment (SMC + Fibroblast, female)
  f) Proteostasis disruption -- chaperone + protein degradation waveforms (female)

Output:
  figures/fig_7_supp.pdf
  figures/fig_7_supp_source_data/
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circadian_utils import (
    apply_style, CLUSTER_CELL_TYPE, ALIGNED_CONDITIONS, MISALIGNED_CONDITIONS,
    CELL_TYPE_COLORS, FIGURES_DIR, ZT_LABELS,
    load_metrics, filter_cyclers, load_waveform_params, sample_waveforms,
    get_non_flat_genes, compute_gene_set_waveform, test_nonzero,
    load_kegg_dict, format_label,
    compute_amplitude, acrophase_rad_to_hours, plot_posterior_violins,
)

apply_style()

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_7_supp_source_data")
NUM_SAMPLES = 5000
NUM_ENRICH_SAMPLES = 100
ZT_HOURS = [0, 6, 12, 18]
SEX = "female"
COND_AL = ALIGNED_CONDITIONS[SEX]
COND_MIS = MISALIGNED_CONDITIONS[SEX]
COND_COLORS = {"Aligned": "#4878CF", "Misaligned": "#D65F5F"}

# ── Step 1: Panel a — amplitude loss ────────────────────────────────────────

print("Step 1: Amplitude loss under misalignment (female)...")

amp_fracs = {}
all_cell_types = list(CLUSTER_CELL_TYPE.values())

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

# ── Step 3: CCR2/CX3CR1 in macrophages ───────────────────────────────────

print("\nStep 3: CCR2/CX3CR1 waveforms in macrophages (female)...")

MAC_GENES = ["Ccr2", "Cx3cr1"]
mac_waveforms = {}

alpha_al_mac, beta_al_mac, mm_al_mac = load_waveform_params("cluster_3", COND_AL)
alpha_mis_mac, beta_mis_mac, mm_mis_mac = load_waveform_params("cluster_3", COND_MIS)

for gene in MAC_GENES:
    mac_waveforms[gene] = {}
    for cond_label, (a, b, m) in [("Aligned", (alpha_al_mac, beta_al_mac, mm_al_mac)),
                                   ("Misaligned", (alpha_mis_mac, beta_mis_mac, mm_mis_mac))]:
        samp, genes = sample_waveforms(a, b, m, [gene], NUM_SAMPLES)
        if genes:
            mac_waveforms[gene][cond_label] = samp[0]
            acro = np.argmax(samp[0].mean(axis=0))
            print(f"  {gene} {cond_label}: peak at ZT{acro * 6}")
        else:
            print(f"  {gene} {cond_label}: NOT FOUND")

# ── Step 4: Clock relative timing in SMCs ─────────────────────────────────

print("\nStep 4: Clock relative timing in female SMCs...")

REFERENCE_GENE = "Arntl"
CLOCK_TARGETS = ["Dbp", "Nr1d1"]
NUM_CLOCK_SAMPLES = 30000


def fft_acrophase_rad(waveform_samples):
    F = np.fft.fft(waveform_samples, axis=-1)
    f1 = F[:, 1]
    return (-np.arctan2(f1.imag, f1.real)) % (2 * np.pi)


def circular_mean(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) % (2 * np.pi)


def signed_circular_diff_scalar(a1, a2):
    return ((a2 - a1 + np.pi) % (2 * np.pi)) - np.pi


clock_shifts = {}

for target in CLOCK_TARGETS:
    clock_shifts[target] = {}
    for cond_label, condition in [("Aligned", COND_AL), ("Misaligned", COND_MIS)]:
        a, b, m = load_waveform_params("cluster_0", condition)
        ref_gene = "Arntl" if "Arntl" in a.index else "Bmal1"
        samp, genes = sample_waveforms(a, b, m, [ref_gene, target], NUM_CLOCK_SAMPLES)
        if len(genes) < 2:
            print(f"  {target} {cond_label}: SKIP")
            continue
        gene_idx = {g: i for i, g in enumerate(genes)}
        ref_rad = fft_acrophase_rad(samp[gene_idx[ref_gene]])
        tgt_rad = fft_acrophase_rad(samp[gene_idx[target]])
        rel_phase = (tgt_rad - ref_rad) % (2 * np.pi)
        clock_shifts[target][cond_label] = rel_phase
        rel_hours = (rel_phase / (2 * np.pi)) * 24
        med = np.median(rel_hours)
        print(f"  {target}-{ref_gene} {cond_label}: median delay = {med:.1f}h")

    if "Aligned" in clock_shifts[target] and "Misaligned" in clock_shifts[target]:
        cm_al = circular_mean(clock_shifts[target]["Aligned"])
        cm_mis = circular_mean(clock_shifts[target]["Misaligned"])
        shift = signed_circular_diff_scalar(cm_al, cm_mis)
        shift_h = (shift / np.pi) * 12.0
        print(f"  {target}: signed relative phase shift = {shift_h:+.1f}h")

# ── Step 5: KEGG enrichment (misaligned vs aligned, female) ──────────────

print("\nStep 5: KEGG waveform enrichment (misaligned vs aligned, female)...")

ENRICH_CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}
MIN_GENES_IN_SET = 3
CREDIBLE_LEVEL = 0.95
KEGG_TOP_N = 8

kegg_dict = load_kegg_dict()
print(f"  KEGG: {len(kegg_dict)} sets")

kegg_enrichment = {}

for cluster, cell_type in ENRICH_CLUSTERS.items():
    metrics_al = load_metrics(cluster, COND_AL)
    metrics_mis = load_metrics(cluster, COND_MIS)
    alpha_al, beta_al, mm_al = load_waveform_params(cluster, COND_AL)
    alpha_mis, beta_mis, mm_mis = load_waveform_params(cluster, COND_MIS)

    non_flat_al = get_non_flat_genes(metrics_al)
    non_flat_mis = get_non_flat_genes(metrics_mis)
    non_flat_union = sorted(non_flat_al | non_flat_mis)

    samp_al, genes_al = sample_waveforms(alpha_al, beta_al, mm_al, non_flat_union, NUM_ENRICH_SAMPLES)
    samp_mis, genes_mis = sample_waveforms(alpha_mis, beta_mis, mm_mis, non_flat_union, NUM_ENRICH_SAMPLES)

    shared_genes = sorted(set(genes_al) & set(genes_mis))
    idx_al_map = {g: i for i, g in enumerate(genes_al)}
    idx_mis_map = {g: i for i, g in enumerate(genes_mis)}
    ord_al = [idx_al_map[g] for g in shared_genes]
    ord_mis = [idx_mis_map[g] for g in shared_genes]

    diff_samples = samp_mis[ord_mis] - samp_al[ord_al]
    gene_index = {g: i for i, g in enumerate(shared_genes)}

    sig_sets = []
    for gs_name, gs_genes in kegg_dict.items():
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
            "gene_set": gs_name,
            "num_genes": len(members),
            "member_genes": ";".join(members),
            "max_amplitude": float(mean_wave.max() - mean_wave.min()),
            "mean_ZT0": float(mean_wave[0]),
            "mean_ZT6": float(mean_wave[1]),
            "mean_ZT12": float(mean_wave[2]),
            "mean_ZT18": float(mean_wave[3]),
        })
    sig_sets.sort(key=lambda x: -x["max_amplitude"])
    kegg_enrichment[cell_type] = pd.DataFrame(sig_sets).head(KEGG_TOP_N)
    print(f"  {cell_type}: {len(sig_sets)} significant KEGG sets, showing top {min(KEGG_TOP_N, len(sig_sets))}")

# ── Step 6: Proteostasis disruption ───────────────────────────────────────

print("\nStep 6: Proteostasis disruption waveforms (female)...")

PROTEO_MAIN = ["Hspa1a", "Xbp1", "Uba52", "Atf3"]

proteo_waveforms = {}

alpha_al_smc, beta_al_smc, mm_al_smc = load_waveform_params("cluster_0", COND_AL)
alpha_mis_smc, beta_mis_smc, mm_mis_smc = load_waveform_params("cluster_0", COND_MIS)

for gene in PROTEO_MAIN:
    proteo_waveforms[gene] = {}
    for cond_label, (a, b, m) in [("Aligned", (alpha_al_smc, beta_al_smc, mm_al_smc)),
                                   ("Misaligned", (alpha_mis_smc, beta_mis_smc, mm_mis_smc))]:
        samp, genes = sample_waveforms(a, b, m, [gene], NUM_SAMPLES)
        if genes:
            proteo_waveforms[gene][cond_label] = samp[0]
        else:
            print(f"  {gene} {cond_label}: NOT FOUND")

found_proteo = [g for g in PROTEO_MAIN if proteo_waveforms[g]]
print(f"  Found waveforms for: {', '.join(found_proteo)}")

# ── Step 7: Export source data ──────────────────────────────────────────────

print("\nStep 7: Exporting source data...")
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

for target, cond_shifts in clock_shifts.items():
    for cond_label, rel_phase in cond_shifts.items():
        rel_hours = (rel_phase / (2 * np.pi)) * 24
        pd.DataFrame({f"{target}_Arntl_delay_hours": rel_hours}).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_c_{target}_{cond_label}.csv"), index=False)

for gene in MAC_GENES:
    for cond_label, samp in mac_waveforms.get(gene, {}).items():
        samp10 = samp / np.log(10)
        pd.DataFrame({"ZT": ZT_LABELS, "mean_log10": samp10.mean(axis=0),
                       "ci_lo_log10": np.quantile(samp10, 0.025, axis=0),
                       "ci_hi_log10": np.quantile(samp10, 0.975, axis=0)}).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_d_{gene}_{cond_label}.csv"), index=False)

for ct, df in kegg_enrichment.items():
    if not df.empty:
        df.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_e_kegg_{ct}.csv"), index=False)

for gene in found_proteo:
    for cond_label, samp in proteo_waveforms[gene].items():
        samp10 = samp / np.log(10)
        pd.DataFrame({"ZT": ZT_LABELS, "mean_log10": samp10.mean(axis=0),
                       "ci_lo_log10": np.quantile(samp10, 0.025, axis=0),
                       "ci_hi_log10": np.quantile(samp10, 0.975, axis=0)}).to_csv(
            os.path.join(SOURCE_DATA_DIR, f"panel_f_{gene}_{cond_label}.csv"), index=False)

print(f"  Source data written to {SOURCE_DATA_DIR}/")

# ── Step 8: Generate figure ─────────────────────────────────────────────────

print("\nStep 8: Generating figure...")

fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
fig.patch.set_facecolor("white")

gs_main = GridSpec(5, 1, figure=fig, hspace=0.50,
                   height_ratios=[0.5, 0.6, 0.45, 0.45, 0.45],
                   left=0.08, right=0.95, top=0.97, bottom=0.03)

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

# ── Row c+d: clock relative timing + macrophage waveforms ─────────────────

gs_cd = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2], wspace=0.45)

for j, target in enumerate(CLOCK_TARGETS):
    ax = fig.add_subplot(gs_cd[0, j])
    for cond_label, color in COND_COLORS.items():
        if cond_label in clock_shifts.get(target, {}):
            rel_hours = (clock_shifts[target][cond_label] / (2 * np.pi)) * 24
            ax.hist(rel_hours, bins=30, density=True, alpha=0.2,
                    color=color, edgecolor="none")
            try:
                sns.kdeplot(rel_hours, ax=ax, color=color, linewidth=1.0,
                            label=cond_label, bw_adjust=0.8, clip=(0, 24))
            except Exception:
                ax.axvline(np.median(rel_hours), color=color, linewidth=1.0,
                           label=cond_label)
    ax.set_xlim(0, 24)
    ax.set_title(f"{target} – Arntl", fontsize=7, fontweight="semibold",
                 color=CELL_TYPE_COLORS["SMC"])
    ax.set_xlabel("Delay (hours)", fontsize=5.5)
    ax.set_ylabel("Density" if j == 0 else "", fontsize=6)
    ax.legend(fontsize=4.5, frameon=False)
    sns.despine(ax=ax)
    if j == 0:
        ax.text(-0.25, 1.15, "c", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

for j, gene in enumerate(MAC_GENES):
    ax = fig.add_subplot(gs_cd[0, 2 + j])
    gene_samps = {cl: s for cl, s in mac_waveforms.get(gene, {}).items()}
    if gene_samps:
        plot_posterior_violins(ax, gene_samps, COND_COLORS)
    ax.set_title(gene, fontsize=7, fontweight="semibold", color=CELL_TYPE_COLORS["Macrophage"])
    ax.set_ylabel("Log10 rate" if j == 0 else "", fontsize=6)
    ax.legend(fontsize=4.5, frameon=False)
    sns.despine(ax=ax)
    if j == 0:
        ax.text(-0.25, 1.15, "d", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# ── Row e: KEGG misalignment enrichment (SMC + Fibroblast) ────────────────

gs_e = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[3], wspace=0.35)

for col_idx, ct in enumerate(["SMC", "Fibroblast"]):
    ax = fig.add_subplot(gs_e[0, col_idx])
    if ct in kegg_enrichment and not kegg_enrichment[ct].empty:
        df = kegg_enrichment[ct]
        zt_cols = ["mean_ZT0", "mean_ZT6", "mean_ZT12", "mean_ZT18"]
        for _, row in df.iterrows():
            vals = [row[c] for c in zt_cols]
            ax.plot(ZT_HOURS, vals, linewidth=1.0, alpha=0.8, label=row["gene_set"])
        ax.axhline(0, color="k", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_xticks(ZT_HOURS)
    ax.set_xticklabels(ZT_LABELS, fontsize=5)
    ax.set_title(ct, fontsize=7, fontweight="semibold",
                 color=CELL_TYPE_COLORS.get(ct, "0.2"))
    ax.set_xlabel("ZT", fontsize=5.5)
    if col_idx == 0:
        ax.set_ylabel("Mis. − Al.\n(log rate)", fontsize=5.5)
    ax.legend(fontsize=3.5, frameon=False, loc="best", ncol=1)
    ax.tick_params(axis="y", labelsize=4.5)
    sns.despine(ax=ax)
    if col_idx == 0:
        ax.text(-0.2, 1.15, "e", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# ── Row f: proteostasis gene waveforms ──────────────────────────────────────

gs_f = GridSpecFromSubplotSpec(1, len(found_proteo), subplot_spec=gs_main[4], wspace=0.35)

for idx, gene in enumerate(found_proteo):
    ax = fig.add_subplot(gs_f[0, idx])
    gene_samps = {cl: s for cl, s in proteo_waveforms[gene].items()}
    if gene_samps:
        plot_posterior_violins(ax, gene_samps, COND_COLORS, width=2.0, alpha=0.5)
    ax.set_title(f"{gene}", fontsize=6.5, fontweight="semibold", color="0.2")
    ax.tick_params(axis="y", labelsize=4.5)
    if idx == 0:
        ax.set_ylabel("Log10 rate", fontsize=5.5)
    if idx < 2:
        ax.legend(fontsize=4, frameon=False)
    sns.despine(ax=ax)
    if idx == 0:
        ax.text(-0.3, 1.2, "f", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_7_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
