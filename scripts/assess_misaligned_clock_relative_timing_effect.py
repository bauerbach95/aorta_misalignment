"""
Assess whether circadian misalignment changes the relative timing between
core clock genes (Arntl vs Dbp, Per2) within SMCs and Fibroblasts.

Computes posterior distributions of acrophase shifts via FFT on 4-ZT
posterior waveform samples, then plots aligned vs misaligned distributions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import circadian_utils as cu

# ── Config ────────────────────────────────────────────────────────────────────

REFERENCE_GENE = "Arntl"  # Bmal1
TARGET_GENES = ["Dbp", "Per1", "Per2", "Nr1d1", "Nr1d2", 'Cry1', 'Cry2']
NUM_SAMPLES = 30000

CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}
SEXES = ["male", "female"]

CONDITION_TEMPLATES = {
    "Aligned": "{sex} aligned bmal1-control",
    "Misaligned": "{sex} misaligned bmal1-control",
}

COND_COLORS = {"Aligned": "#4878CF", "Misaligned": "#D65F5F"}

OUT_DIR = os.path.join(cu.FIGURES_DIR, "clock_relative_timing")


# ── Acrophase via FFT ────────────────────────────────────────────────────────

def fft_acrophase_hours(waveform_samples):
    """Compute acrophase in hours for each posterior sample.

    Args:
        waveform_samples: np.ndarray [num_samples, 4] in log-rate space

    Returns:
        np.ndarray [num_samples] of peak hours in [0, 24)
    """
    F = np.fft.fft(waveform_samples, axis=-1)
    f1 = F[:, 1]  # 24h component
    phi = (-np.arctan2(f1.imag, f1.real)) % (2 * np.pi)
    return phi * 24 / (2 * np.pi)


def circular_shift(target_hours, ref_hours):
    """Compute circular phase delay (target - ref) wrapped to [0, 24)."""
    return (target_hours - ref_hours) % 24


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cu.apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Resolve reference gene name (might be "Bmal1" in some datasets)
    ref_gene_aliases = ["Arntl", "Bmal1"]

    for cluster, cell_type in CLUSTERS.items():
        fig, axes = plt.subplots(
            len(TARGET_GENES), len(SEXES),
            figsize=(4.5, 7.5), squeeze=False,
        )

        for col, sex in enumerate(SEXES):
            for row, target in enumerate(TARGET_GENES):
                ax = axes[row, col]

                for cond_label, cond_template in CONDITION_TEMPLATES.items():
                    condition = cond_template.format(sex=sex)

                    # Load waveform params
                    try:
                        alpha_df, beta_df, minmax_df = cu.load_waveform_params(
                            cluster, condition
                        )
                    except FileNotFoundError:
                        print(f"  SKIP {cell_type} {sex} {cond_label}: data not found")
                        continue

                    # Resolve reference gene name
                    ref_gene = None
                    for alias in ref_gene_aliases:
                        if alias in alpha_df.index:
                            ref_gene = alias
                            break
                    if ref_gene is None:
                        print(f"  SKIP {cell_type} {sex} {cond_label}: "
                              f"reference gene not found ({ref_gene_aliases})")
                        continue

                    # Check target gene exists
                    if target not in alpha_df.index:
                        print(f"  SKIP {cell_type} {sex} {cond_label}: "
                              f"{target} not found")
                        continue

                    # Sample waveforms
                    gene_list = [ref_gene, target]
                    samples, genes = cu.sample_waveforms(
                        alpha_df, beta_df, minmax_df, gene_list,
                        num_samples=NUM_SAMPLES,
                    )
                    if len(genes) < 2:
                        print(f"  SKIP {cell_type} {sex} {cond_label}: "
                              f"insufficient genes sampled")
                        continue

                    gene_idx = {g: i for i, g in enumerate(genes)}
                    ref_waveforms = samples[gene_idx[ref_gene]]    # [N, 4]
                    target_waveforms = samples[gene_idx[target]]   # [N, 4]

                    # Compute acrophases
                    ref_hours = fft_acrophase_hours(ref_waveforms)
                    target_hours = fft_acrophase_hours(target_waveforms)

                    # Compute phase shift
                    shifts = circular_shift(target_hours, ref_hours)

                    # Plot KDE
                    ax.hist(
                        shifts, bins=30, density=True, alpha=0.25,
                        color=COND_COLORS[cond_label], edgecolor="none",
                    )
                    try:
                        sns.kdeplot(
                            shifts, ax=ax, color=COND_COLORS[cond_label],
                            linewidth=1.0, label=cond_label, bw_adjust=0.8,
                            clip=(0, 24),
                        )
                    except Exception:
                        ax.axvline(
                            np.median(shifts), color=COND_COLORS[cond_label],
                            linewidth=1.0, label=cond_label,
                        )

                    # Print stats
                    med = np.median(shifts)
                    R = np.abs(np.mean(np.exp(1j * shifts * 2 * np.pi / 24)))
                    circ_std = np.sqrt(-2 * np.log(max(R, 1e-10))) * 24 / (2 * np.pi)
                    print(f"  {cell_type} | {sex:6s} | {cond_label:10s} | "
                          f"{target}-{REFERENCE_GENE}: "
                          f"median={med:+.1f}h, circ_std={circ_std:.1f}h")

                # Formatting
                ax.set_xlim(0, 24)
                ax.set_xlabel("Delay relative to Arntl (hours)" if row == len(TARGET_GENES) - 1 else "")
                ax.set_ylabel("Density" if col == 0 else "")
                title = f"{target} – {REFERENCE_GENE}"
                if row == 0:
                    title = f"{sex.capitalize()}\n{title}"
                ax.set_title(title, fontsize=7)
                ax.legend(fontsize=5, frameon=False)
                sns.despine(ax=ax)

        fig.suptitle(f"{cell_type}: Clock Gene Phase Shifts", fontsize=9,
                     fontweight="semibold", y=1.02)
        fig.tight_layout()

        out_path = os.path.join(OUT_DIR, f"{cell_type}_clock_phase_shifts.pdf")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
