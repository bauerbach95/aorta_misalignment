"""
Compare aligned vs misaligned conditions for shared confident cyclers.

For SMC and Fibroblast:
1. Acrophase scatter: aligned (x) vs misaligned (y) peak times
2. Amplitude shift: histogram of P(amp_misaligned > amp_aligned) across genes
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circadian_utils as cu

# ── Config ────────────────────────────────────────────────────────────────────

CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}
SEXES = ["male", "female"]
SEX_MARKERS = {"male": "o", "female": "D"}
SEX_COLORS = {"male": "#5A7EBD", "female": "#C45B5B"}
NUM_SAMPLES = 5000

OUT_DIR = os.path.join(cu.FIGURES_DIR, "misalignment_shared_cyclers")


def get_acrophase_hours(metrics_df, genes):
    """Get MAP acrophase in hours [0, 24) for a set of genes."""
    acro_rad = metrics_df.loc[genes, "expected_acrophase"].values.astype(float)
    return (acro_rad / (2 * np.pi) * 24) % 24


def compute_amplitude(samples):
    """Compute amplitude (max - min over ZTs) per sample.

    Args:
        samples: np.ndarray [num_genes, num_samples, 4] in log-rate space

    Returns:
        np.ndarray [num_genes, num_samples]
    """
    return samples.max(axis=-1) - samples.min(axis=-1)


def main():
    cu.apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.5))

    for col, (cluster, cell_type) in enumerate(CLUSTERS.items()):
        ax_acro = axes[0, col]
        ax_amp = axes[1, col]

        sex_fracs = {}  # {sex: array of fractions}

        for sex in SEXES:
            cond_aligned = f"{sex} aligned bmal1-control"
            cond_misaligned = f"{sex} misaligned bmal1-control"

            # Load metrics and find cyclers in each condition
            try:
                metrics_al = cu.load_metrics(cluster, cond_aligned)
                metrics_mis = cu.load_metrics(cluster, cond_misaligned)
            except FileNotFoundError:
                print(f"  SKIP {cell_type} {sex}: data not found")
                continue

            cyclers_al = cu.filter_cyclers(metrics_al)
            cyclers_mis = cu.filter_cyclers(metrics_mis)
            shared = sorted(cyclers_al & cyclers_mis)

            print(f"  {cell_type} {sex}: {len(cyclers_al)} aligned, "
                  f"{len(cyclers_mis)} misaligned, {len(shared)} shared")

            if not shared:
                continue

            # ── Acrophase scatter ─────────────────────────────────────
            acro_al = get_acrophase_hours(metrics_al, shared)
            acro_mis = get_acrophase_hours(metrics_mis, shared)

            ax_acro.scatter(
                acro_al, acro_mis, s=8, alpha=0.5,
                marker=SEX_MARKERS[sex], color=SEX_COLORS[sex],
                edgecolors="none", label=f"{sex} (n={len(shared)})",
            )

            # ── Amplitude comparison ──────────────────────────────────
            alpha_al, beta_al, mm_al = cu.load_waveform_params(cluster, cond_aligned)
            alpha_mis, beta_mis, mm_mis = cu.load_waveform_params(cluster, cond_misaligned)

            samp_al, genes_al = cu.sample_waveforms(
                alpha_al, beta_al, mm_al, shared, num_samples=NUM_SAMPLES,
            )
            samp_mis, genes_mis = cu.sample_waveforms(
                alpha_mis, beta_mis, mm_mis, shared, num_samples=NUM_SAMPLES,
            )

            # Align gene order
            gene_idx_al = {g: i for i, g in enumerate(genes_al)}
            gene_idx_mis = {g: i for i, g in enumerate(genes_mis)}
            common_genes = [g for g in shared if g in gene_idx_al and g in gene_idx_mis]
            idx_al = [gene_idx_al[g] for g in common_genes]
            idx_mis = [gene_idx_mis[g] for g in common_genes]

            amp_al = compute_amplitude(samp_al[idx_al])    # [n_genes, N]
            amp_mis = compute_amplitude(samp_mis[idx_mis])  # [n_genes, N]

            # P(misaligned amplitude > aligned amplitude) per gene
            frac_larger = (amp_mis > amp_al).mean(axis=1)
            sex_fracs[sex] = frac_larger

            print(f"    Amplitude: median P(mis>al) = {np.median(frac_larger):.2f}")

        # ── Format acrophase scatter ──────────────────────────────────
        ax_acro.plot([0, 24], [0, 24], "k--", linewidth=0.5, alpha=0.4)
        ax_acro.set_xlim(0, 24)
        ax_acro.set_ylim(0, 24)
        ax_acro.set_xticks([0, 6, 12, 18, 24])
        ax_acro.set_yticks([0, 6, 12, 18, 24])
        ax_acro.set_xlabel("Aligned acrophase (h)")
        ax_acro.set_ylabel("Misaligned acrophase (h)" if col == 0 else "")
        ax_acro.set_title(cell_type, fontsize=8, fontweight="semibold")
        ax_acro.legend(fontsize=5, frameon=False, loc="upper left")
        ax_acro.set_aspect("equal")
        sns.despine(ax=ax_acro)

        # ── Format amplitude density ──────────────────────────────────
        ax_amp.axvline(0.5, color="k", linestyle="--", linewidth=0.6, alpha=0.5)
        for sex, fracs in sex_fracs.items():
            sns.kdeplot(
                fracs, ax=ax_amp, color=SEX_COLORS[sex],
                linewidth=1.0, label=f"{sex} (n={len(fracs)})",
                clip=(0, 1), bw_adjust=0.8,
            )
        ax_amp.set_xlim(0, 1)
        ax_amp.set_xlabel("P(amplitude misaligned > aligned)")
        ax_amp.set_ylabel("Density" if col == 0 else "")
        ax_amp.legend(fontsize=5, frameon=False)
        ax_amp.set_title(cell_type, fontsize=8, fontweight="semibold")
        sns.despine(ax=ax_amp)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "shared_cycler_acrophase_amplitude.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
