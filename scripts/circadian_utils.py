"""
Shared utilities for circadian rhythm analysis and figure generation.

Used by gen_fig_2.py and supplementary figure scripts.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import circmean
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/Users/mingyaolab/Dropbox/aorta_circadian_data"
NONPARAM_REG_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05/nonparametric_reg",
)
GENE_SETS_DIR = os.path.join(DATA_ROOT, "gene_sets")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

# ── Cell types & conditions ──────────────────────────────────────────────────

CLUSTER_CELL_TYPE = {
    "cluster_0": "SMC",
    "cluster_1": "Fibroblast",
    "cluster_2": "EC",
    "cluster_3": "Macrophage",
}

ALIGNED_CONDITIONS = {
    "male": "male aligned bmal1-control",
    "female": "female aligned bmal1-control",
}

CELL_TYPE_COLORS = {
    "SMC": "#E64B35",
    "Fibroblast": "#4DBBD5",
    "EC": "#00A087",
    "Macrophage": "#3C5488",
}

# ── Cycler thresholds ────────────────────────────────────────────────────────

FRAC_CELL_DETECTED_MIN = 0.01
NUM_CELL_DETECTED_MIN = 50
EXPECTED_MESOR_MIN = -13.8155
FRAC_CIRC_LARGEST_COMP_MIN = 0.6
WAVEFORM_BF_MIN = 2.0
ACROPHASE_HOUR_THRESH = 3.0

# ── Enrichment settings ─────────────────────────────────────────────────────

MIN_GENES_IN_PATHWAY = 6
NUM_ACROPHASE_BINS = 12
LRT_THRESHOLD = 2.0
TOP_N_PATHWAYS = 10

# ── KEGG exclusion list ─────────────────────────────────────────────────────

KEGG_EXCLUDE_PATTERNS = [
    r"cancer", r"carcinoma", r"leukemia", r"melanoma", r"glioma", r"lymphoma",
    r"papillomavirus", r"herpes", r"hepatitis", r"influenza", r"HIV", r"HTLV",
    r"measles", r"Epstein", r"virus", r"infection", r"Vibrio", r"Salmonella",
    r"Shigella", r"Legionella", r"Staphylococcus", r"Tuberculosis", r"Malaria",
    r"Chagas", r"Leishmaniasis", r"Toxoplasmosis", r"Amoebiasis",
    r"Parkinson", r"Alzheimer", r"Huntington", r"Prion",
    r"Cushing", r"diabetic complication",
    r"MicroRNAs in",
]

# ── Matplotlib styling ───────────────────────────────────────────────────────

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.weight": "regular",
    "mathtext.default": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.titlesize": 8,
    "lines.linewidth": 0.8,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.transparent": False,
}


def apply_style():
    matplotlib.rcParams.update(RC_PARAMS)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_metrics(cluster, condition):
    path = os.path.join(NONPARAM_REG_DIR, cluster, condition, "de_novo_metrics.tsv")
    return pd.read_table(path, sep="\t", index_col="gene")


def filter_cyclers(metric_df):
    mask = (
        (metric_df["frac_cell_detected"] > FRAC_CELL_DETECTED_MIN)
        & (metric_df["num_cell_detected"] > NUM_CELL_DETECTED_MIN)
        & (metric_df["expected_mesor"] >= EXPECTED_MESOR_MIN)
        & (metric_df["frac_circadian_samples_largest_component"] >= FRAC_CIRC_LARGEST_COMP_MIN)
        & (metric_df["waveform_over_circadian_component_subtracted_log10_bf"] >= WAVEFORM_BF_MIN)
    )
    return set(metric_df.index[mask])


# ── Circular math ────────────────────────────────────────────────────────────


def acrophase_rad_to_hours(rad):
    return (rad / (2 * np.pi)) * 24.0


def circular_hour_distance(h1, h2, period=24.0):
    d = abs(h1 - h2) % period
    return np.minimum(d, period - d)


# ── Gene set parsing ─────────────────────────────────────────────────────────


def parse_gsea_set_file(path):
    gsea_dict = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            name = parts[0]
            genes = [g.lower().capitalize() for g in parts[2:] if g.strip()]
            if genes:
                gsea_dict[name] = genes
    return gsea_dict


def clean_tf_name(name):
    return re.sub(r"\s*\((?:human|mouse)\)\s*", "", name).strip()


def filter_kegg_pathways(kegg_dict):
    pattern = re.compile("|".join(KEGG_EXCLUDE_PATTERNS), re.IGNORECASE)
    return {k: v for k, v in kegg_dict.items() if not pattern.search(k)}


def load_kegg_dict():
    raw = parse_gsea_set_file(os.path.join(GENE_SETS_DIR, "KEGG_2019_Mouse.txt"))
    filtered = filter_kegg_pathways(raw)
    print(f"  {len(raw)} total KEGG pathways -> {len(filtered)} after filtering")
    return filtered


def load_tf_dict():
    return parse_gsea_set_file(os.path.join(GENE_SETS_DIR, "TRANSFAC_and_JASPAR_PWMs.txt"))


# ── Enrichment ───────────────────────────────────────────────────────────────


def compute_acrophase_enrichment(gene_acrophases_rad, gene_set_dict, num_bins=NUM_ACROPHASE_BINS):
    gene_set_names = []
    num_genes_list = []
    member_genes_list = []
    lrt_mat = []
    peak_hours_list = []
    circ_std_list = []

    for gs_name, gs_genes in gene_set_dict.items():
        member_genes = [g for g in gs_genes if g in gene_acrophases_rad]
        if len(member_genes) < MIN_GENES_IN_PATHWAY:
            continue
        acrophases = np.array([gene_acrophases_rad[g] for g in member_genes])
        bin_indices = np.floor((acrophases / (2 * np.pi)) * num_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        bin_counts = np.bincount(bin_indices, minlength=num_bins).astype(float)
        bin_density = bin_counts / bin_counts.sum()
        uniform_val = 1.0 / num_bins
        lrt = bin_density / uniform_val

        circ_mean = circmean(acrophases, high=2 * np.pi, low=0)
        R = np.abs(np.mean(np.exp(1j * acrophases)))
        circ_std_h = acrophase_rad_to_hours(np.sqrt(-2 * np.log(max(R, 1e-10))))

        gene_set_names.append(gs_name)
        num_genes_list.append(len(member_genes))
        member_genes_list.append(";".join(member_genes))
        lrt_mat.append(lrt)
        peak_hours_list.append(acrophase_rad_to_hours(circ_mean))
        circ_std_list.append(circ_std_h)

    if not gene_set_names:
        return pd.DataFrame(), pd.DataFrame()

    lrt_mat = np.array(lrt_mat)
    enrich_df = pd.DataFrame({
        "gene_set": gene_set_names,
        "num_genes": num_genes_list,
        "member_genes": member_genes_list,
        "max_lrt": np.max(lrt_mat, axis=1),
        "peak_bin": np.argmax(lrt_mat, axis=1),
        "circ_mean_hour": peak_hours_list,
        "circ_std_hour": circ_std_list,
    })
    enrich_df["peak_hour"] = (enrich_df["peak_bin"] + 0.5) * (24.0 / num_bins)
    enrich_df = enrich_df.set_index("gene_set").sort_values("max_lrt", ascending=False)

    lrt_df = pd.DataFrame(lrt_mat, index=gene_set_names,
                           columns=[f"bin_{i}_ZT{i*2}-{i*2+2}" for i in range(num_bins)])
    lrt_df = lrt_df.loc[enrich_df.index]

    return enrich_df, lrt_df


def run_enrichment(acrophases, gene_set_dict, label):
    """Run enrichment, filter by LRT threshold, return (full_df, top_df, lrt_df)."""
    enrich_df, lrt_df = compute_acrophase_enrichment(acrophases, gene_set_dict)
    if not enrich_df.empty:
        enrich_df = enrich_df[enrich_df["max_lrt"] >= LRT_THRESHOLD]
    n = len(enrich_df) if not enrich_df.empty else 0
    top_df = enrich_df.head(TOP_N_PATHWAYS)
    print(f"  {label}: {n} enriched (showing top {min(n, TOP_N_PATHWAYS)})")
    return enrich_df, top_df, lrt_df


# ── Plotting ─────────────────────────────────────────────────────────────────

BIN_WIDTH = 2 * np.pi / NUM_ACROPHASE_BINS


def format_label(s, max_len=32, is_tf=False):
    if is_tf:
        s = clean_tf_name(s)
        s = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "\u2026"


def make_rose_plot(ax, enrich_df, lrt_df, title, is_tf=False):
    """Rose plot with subdivided wedge bars for pathways/TFs."""
    num_bins = NUM_ACROPHASE_BINS
    zt_ticks = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(zt_ticks)
    ax.set_xticklabels([f"ZT{h}" for h in range(0, 24, 2)],
                       fontsize=5.5, fontweight="medium")
    ax.tick_params(axis="y", labelsize=4.5, pad=1)
    ax.tick_params(axis="x", pad=3)
    ax.grid(True, alpha=0.2, linewidth=0.3, color="0.5")
    ax.spines["polar"].set_linewidth(0.4)
    ax.spines["polar"].set_color("0.7")
    ax.set_title(title, fontsize=8, fontweight="semibold", pad=16, color="0.15")

    if enrich_df.empty:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        return

    pathways = list(enrich_df.index)
    n_pw = len(pathways)
    if n_pw <= 8:
        colors = plt.cm.Set2(np.linspace(0, 1, 8))[:n_pw]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_pw]

    bin_to_pathways = defaultdict(list)
    for i, pw in enumerate(pathways):
        peak_bin = int(enrich_df.loc[pw, "peak_bin"])
        bin_to_pathways[peak_bin].append((i, pw))

    max_lrt_val = 0
    for peak_bin, pw_list in bin_to_pathways.items():
        n_in_bin = len(pw_list)
        sub_width = BIN_WIDTH * 0.85 / n_in_bin
        bin_start = zt_ticks[peak_bin] + BIN_WIDTH * 0.075
        for j, (i, pw) in enumerate(pw_list):
            max_lrt = enrich_df.loc[pw, "max_lrt"]
            max_lrt_val = max(max_lrt_val, max_lrt)
            theta_center = bin_start + j * sub_width + sub_width / 2
            ax.bar(theta_center, max_lrt, width=sub_width * 0.9, bottom=0,
                   color=colors[i], alpha=0.88, edgecolor="white",
                   linewidth=0.4, zorder=3)

    ref_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ref_theta, np.ones_like(ref_theta), color="0.55",
            linewidth=0.5, linestyle="--", alpha=0.6, zorder=2)
    ax.set_ylim(0, max(max_lrt_val * 1.18, 2.5))

    labels = [format_label(pw, is_tf=is_tf) for pw in pathways]
    handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor="0.85", linewidth=0.3,
                       label=labels[i])
        for i in range(n_pw)
    ]
    leg = ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        fontsize=4.5, frameon=False, handlelength=0.7, handleheight=0.7,
        labelspacing=0.25, ncol=2 if n_pw > 5 else 1, columnspacing=0.8,
    )
    for text in leg.get_texts():
        text.set_color("0.2")


# ── Source data export helpers ───────────────────────────────────────────────


def export_enrichment_tables(source_dir, prefix, enrichments_full, label_col="pathway"):
    """Export enrichment + LRT bin tables for each cell type."""
    for ct, (enrich_df, lrt_df) in enrichments_full.items():
        if not enrich_df.empty:
            out = enrich_df.reset_index().rename(columns={"gene_set": label_col})
            out.to_csv(os.path.join(source_dir, f"{prefix}_enrichment_{ct}.csv"), index=False)
            lrt_df.to_csv(os.path.join(source_dir, f"{prefix}_lrt_bins_{ct}.csv"))
