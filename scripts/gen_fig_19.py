#!/usr/bin/env python3
"""Supplementary Figure 19: WT vs iKO cycler counts,
controlled for cell counts and library sizes.

Panel a: Male aligned WT vs KO — cycler count vs BF threshold (SMC, Fibroblast)
Panel b: Female aligned WT vs KO — same

Uses downsampled data (matched cell counts and library sizes across conditions).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from circadian_utils import (
    DATA_ROOT, FIGURES_DIR,
    FRAC_CELL_DETECTED_MIN, NUM_CELL_DETECTED_MIN,
    FRAC_CIRC_LARGEST_COMP_MIN, WAVEFORM_BF_MIN,
    CELL_TYPE_COLORS, apply_style,
)

apply_style()

DOWNSAMPLED_KO_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05/male_cell_type_rhythmicity_same_lib_size_and_cell_count",
)

CLUSTERS = {"cluster_0": "SMC", "cluster_1": "Fibroblast"}

CONDITIONS = {
    "male": {
        "wt": "male aligned bmal1-control",
        "ko": "male aligned bmal1-ko",
    },
    "female": {
        "wt": "female aligned bmal1-control",
        "ko": "female aligned bmal1-ko",
    },
}

SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_19_source_data")


def load_metrics(cluster, condition):
    path = os.path.join(DOWNSAMPLED_KO_DIR, cluster, condition, "de_novo_metrics.tsv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Downsampled data missing: {path}\n"
            f"Run create_downsampled_fibroblast_ko_data.py first."
        )
    return pd.read_table(path, sep="\t", index_col="gene")


def filter_cyclers(metric_df, bf_threshold=WAVEFORM_BF_MIN):
    mask = (
        (metric_df["frac_cell_detected"] > FRAC_CELL_DETECTED_MIN)
        & (metric_df["num_cell_detected"] > NUM_CELL_DETECTED_MIN)
        & (metric_df["frac_circadian_samples_largest_component"] >= FRAC_CIRC_LARGEST_COMP_MIN)
        & (metric_df["waveform_over_circadian_component_subtracted_log10_bf"] >= bf_threshold)
    )
    return set(metric_df.index[mask])


bf_thresholds = np.concatenate([np.arange(0.5, 3.0, 0.25), np.arange(3, 11, 1.0)])

print("Computing rhythmicity sweeps (WT vs iKO, downsampled)...")

sweep_data = {}
for sex, conds in CONDITIONS.items():
    sex_data = {}
    for cluster, cell_type in CLUSTERS.items():
        wt_df = load_metrics(cluster, conds["wt"])
        ko_df = load_metrics(cluster, conds["ko"])

        results = []
        for thresh in bf_thresholds:
            wt_cyc = filter_cyclers(wt_df, bf_threshold=thresh)
            ko_cyc = filter_cyclers(ko_df, bf_threshold=thresh)
            shared = wt_cyc & ko_cyc
            results.append({
                "bf_threshold": thresh,
                "wt": len(wt_cyc),
                "ko": len(ko_cyc),
                "shared": len(shared),
                "unique_wt": len(wt_cyc - ko_cyc),
                "unique_ko": len(ko_cyc - wt_cyc),
            })
        sex_data[cell_type] = pd.DataFrame(results)
        print(f"  {sex} {cell_type}: WT={results[0]['wt']}, KO={results[0]['ko']} at BF>={bf_thresholds[0]}")
    sweep_data[sex] = sex_data

# ── Export source data ─────────────────────────────────────────────────────

os.makedirs(SOURCE_DATA_DIR, exist_ok=True)
for sex, sex_data in sweep_data.items():
    panel = "a" if sex == "male" else "b"
    for ct, df in sex_data.items():
        df.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_{panel}_sweep_{sex}_{ct}.csv"), index=False)
print(f"Source data written to {SOURCE_DATA_DIR}/")

# ── Generate figure ────────────────────────────────────────────────────────

fig = plt.figure(figsize=(6.0, 5.0), dpi=300)
fig.patch.set_facecolor("white")

gs_main = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35,
                   left=0.10, right=0.95, top=0.93, bottom=0.08)

line_colors = {"wt": "#2C7BB6", "ko": "#D7191C", "shared": "#9B59B6"}
panel_labels = {"male": "a", "female": "b"}
cell_types_ordered = ["SMC", "Fibroblast"]

for row_idx, (sex, sex_data) in enumerate(sweep_data.items()):
    for col_idx, ct in enumerate(cell_types_ordered):
        ax = fig.add_subplot(gs_main[row_idx, col_idx])
        df = sex_data[ct]

        ax.plot(df["bf_threshold"], df["wt"], color=line_colors["wt"],
                linewidth=1.2, label="Aligned WT")
        ax.plot(df["bf_threshold"], df["ko"], color=line_colors["ko"],
                linewidth=1.2, label="Aligned iKO")
        ax.plot(df["bf_threshold"], df["shared"], color=line_colors["shared"],
                linewidth=1.0, linestyle="--", label="Shared", alpha=0.7)

        ax.set_title(f"{ct} ({sex})", fontsize=7, fontweight="semibold",
                     color=CELL_TYPE_COLORS[ct])
        ax.set_xlabel(r"$\log_{10}$ circadian Bayes factor threshold", fontsize=6)
        if col_idx == 0:
            ax.set_ylabel("Number of cyclers", fontsize=6.5)
        ax.tick_params(axis="both", labelsize=5.5)
        ax.legend(fontsize=5, frameon=False, loc="upper right")
        sns.despine(ax=ax)

        if col_idx == 0:
            ax.text(-0.18, 1.12, panel_labels[sex], transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")

os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_19_supp.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
