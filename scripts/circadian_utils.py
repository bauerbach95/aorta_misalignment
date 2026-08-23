"""
Shared utilities for circadian rhythm analysis and figure generation.

Used by gen_fig_2.py and supplementary figure scripts.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
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
DOWNSAMPLED_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05/male_vs_female_aligned_cell_type_rhythmicity",
)
GENE_SETS_DIR = os.path.join(DATA_ROOT, "gene_sets")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
SMC_SWITCHING_DIR = os.path.join(
    "/Users/mingyaolab/Dropbox/athero_data/preprocessed",
    "RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_"
    "scale_True_zero-centered_True_scale-per-batch_False_"
    "sample-min-disp_3.0_zsgreen-min_disp_3.0_all-min-disp_0.7",
    "desc_results",
    "subject_RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_"
    "dims_128_resolution_0.35_tol_0.005_run_0",
    "pseudotime_X_velocity_results",
    "pseudotime-col_pseudotime_2_batch-subset_['RPS007']_"
    "num-bins_4_bin-subset_[0, 1, 2, 3]_p-val_thresh_0.05",
    "gene_patterns",
)

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

MISALIGNED_CONDITIONS = {
    "male": "male misaligned bmal1-control",
    "female": "female misaligned bmal1-control",
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
    r"cancer", r"carcinoma", r"carcinogen", r"leukemia", r"melanoma", r"glioma",
    r"lymphoma", r"sarcoma", r"tumor",
    r"papillomavirus", r"herpes", r"hepatitis", r"influenza", r"HIV", r"HTLV",
    r"measles", r"Epstein", r"virus", r"infection", r"invasion",
    r"Vibrio", r"Salmonella", r"Shigella", r"Legionella", r"Legionellosis",
    r"Staphylococcus", r"Tuberculosis", r"Malaria", r"Pertussis",
    r"Chagas", r"Leishmaniasis", r"Toxoplasmosis", r"Amoebiasis",
    r"trypanosomiasis",
    r"arthritis", r"asthma", r"lupus",
    r"Parkinson", r"Alzheimer", r"Huntington", r"Prion",
    r"Cushing", r"diabetic complication",
    r"MicroRNAs in",
]

# ── Reactome exclusion list ───────────────────────────────────────────────

REACTOME_EXCLUDE_PATTERNS = [
    r"DISEASE", r"CANCER", r"CARCINOMA", r"LEUKEMIA", r"MELANOMA", r"GLIOMA",
    r"LYMPHOMA", r"SARCOMA", r"TUMOR",
    r"INFECTION", r"VIRUS", r"INFLUENZA", r"HIV", r"HEPATITIS",
    r"TUBERCULOSIS", r"MALARIA",
    r"DEFECT", r"ABNORMAL",
    r"OLFACTORY", r"TASTE",
    r"SARS_COV", r"CORONA",
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


# ── Waveform sampling ────────────────────────────────────────────────────────

ZT_LABELS = ["ZT0", "ZT6", "ZT12", "ZT18"]


def load_waveform_params(cluster, condition):
    """Load alpha, beta, min/max parameter DataFrames for a group."""
    base = os.path.join(NONPARAM_REG_DIR, cluster, condition)
    alpha_df = pd.read_table(os.path.join(base, "gene_log_alpha.tsv"), sep="\t", index_col="gene")
    beta_df = pd.read_table(os.path.join(base, "gene_log_beta.tsv"), sep="\t", index_col="gene")
    minmax_df = pd.read_table(os.path.join(base, "log_min_max.tsv"), sep="\t", index_col="gene")
    return alpha_df, beta_df, minmax_df


def get_non_flat_genes(metrics_df, bf_threshold=2.0, frac_min=0.01):
    """Return set of genes whose waveform is sufficiently non-flat and well-detected.

    bf_threshold: log10 Bayes factor (waveform vs flat). 2.0 = 100x better.
    """
    log10_bf = (
        (metrics_df["waveform_log_evidence"] - metrics_df["flat_waveform_log_evidence"])
        / np.log(10)
    )
    mask = (log10_bf >= bf_threshold) & (metrics_df["frac_cell_detected"] >= frac_min)
    return set(metrics_df.index[mask])


def sample_waveforms(alpha_df, beta_df, minmax_df, gene_list, num_samples=100):
    """Sample posterior waveforms for a list of genes.

    Returns:
        samples: np.ndarray [num_genes, num_samples, 4] in log-rate space
        genes: list of gene names in the same order as axis 0
    """
    genes = [g for g in gene_list if g in alpha_df.index]
    if not genes:
        return np.empty((0, num_samples, 4)), []

    alpha = np.exp(alpha_df.loc[genes].values)   # [num_genes, 4]
    beta = np.exp(beta_df.loc[genes].values)     # [num_genes, 4]
    log_min = minmax_df.loc[genes, "log_min"].values  # [num_genes]
    log_max = minmax_df.loc[genes, "log_max"].values  # [num_genes]

    dist = torch.distributions.Beta(
        torch.tensor(alpha.T, dtype=torch.float32),  # [4, num_genes]
        torch.tensor(beta.T, dtype=torch.float32),
    )
    raw = dist.sample((num_samples,)).numpy()  # [num_samples, 4, num_genes] in [0,1]

    # Rescale to log-rate space
    log_min_bc = log_min[np.newaxis, np.newaxis, :]  # [1, 1, num_genes]
    log_max_bc = log_max[np.newaxis, np.newaxis, :]
    raw = raw * (log_max_bc - log_min_bc) + log_min_bc  # [num_samples, 4, num_genes]

    # Transpose to [num_genes, num_samples, 4]
    samples = raw.transpose(2, 0, 1)
    return samples, genes


def compute_gene_set_waveform(all_samples, gene_index, gene_set_genes, mode="mesor_subtract"):
    """Compute a gene-set-level posterior waveform from pre-sampled gene waveforms.

    Args:
        all_samples: np.ndarray [num_all_genes, num_samples, 4]
        gene_index: dict mapping gene name -> index into axis 0 of all_samples
        gene_set_genes: list of gene names in this set
        mode: "mesor_subtract" (single group) or "average" (pre-computed diffs)

    Returns:
        gene_set_waveform: np.ndarray [num_samples, 4] or None if too few genes
    """
    idxs = [gene_index[g] for g in gene_set_genes if g in gene_index]
    if not idxs:
        return None
    sub = all_samples[idxs]  # [n_genes, num_samples, 4]

    if mode == "mesor_subtract":
        mesor = sub.mean(axis=2, keepdims=True)  # [n_genes, num_samples, 1]
        sub = sub - mesor

    return sub.mean(axis=0)  # [num_samples, 4]


def test_time_varying(gene_set_waveform, credible_level=0.95):
    """Test if a gene-set waveform varies significantly over 4 time points.

    Uses pairwise comparisons with Bonferroni correction (6 pairs).

    Returns:
        is_significant: bool
        pairwise_results: list of dicts
    """
    n_pairs = 6
    alpha_corrected = (1 - credible_level) / n_pairs
    lo_q = alpha_corrected / 2
    hi_q = 1 - alpha_corrected / 2

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    is_significant = False
    pairwise_results = []

    for i, j in pairs:
        diff = gene_set_waveform[:, i] - gene_set_waveform[:, j]
        ci_lo = np.quantile(diff, lo_q)
        ci_hi = np.quantile(diff, hi_q)
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        if excludes_zero:
            is_significant = True
        pairwise_results.append({
            "pair": f"{ZT_LABELS[i]} - {ZT_LABELS[j]}",
            "mean_diff": float(np.mean(diff)),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "excludes_zero": excludes_zero,
        })

    return is_significant, pairwise_results


def test_nonzero(gene_set_waveform, credible_level=0.95):
    """Test if a waveform is significantly different from zero at any time point.

    Useful for two-group difference waveforms: identifies gene sets where
    the groups differ at specific ZTs. Bonferroni-corrected over 4 time points.

    Returns:
        is_significant: bool (True if any ZT excludes zero)
        pointwise_results: list of dicts with per-ZT CI info
    """
    n_tests = 4
    alpha_corrected = (1 - credible_level) / n_tests
    lo_q = alpha_corrected / 2
    hi_q = 1 - alpha_corrected / 2

    is_significant = False
    pointwise_results = []

    for k in range(4):
        vals = gene_set_waveform[:, k]
        ci_lo = np.quantile(vals, lo_q)
        ci_hi = np.quantile(vals, hi_q)
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        if excludes_zero:
            is_significant = True
        pointwise_results.append({
            "zt": ZT_LABELS[k],
            "mean": float(np.mean(vals)),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "excludes_zero": excludes_zero,
        })

    return is_significant, pointwise_results


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


def clean_reactome_name(name):
    """REACTOME_SMOOTH_MUSCLE_CONTRACTION -> Smooth muscle contraction."""
    name = name.replace("REACTOME_", "")
    name = name.replace("_", " ")
    return name[0].upper() + name[1:].lower() if name else name


def filter_reactome_pathways(d, max_genes=500):
    pattern = re.compile("|".join(REACTOME_EXCLUDE_PATTERNS), re.IGNORECASE)
    return {k: v for k, v in d.items() if not pattern.search(k) and len(v) <= max_genes}


def load_reactome_dict():
    raw = parse_gsea_set_file(
        os.path.join(GENE_SETS_DIR, "gsea_reactome_pathways_gene_sets")
    )
    filtered = filter_reactome_pathways(raw)
    print(f"  {len(raw)} total Reactome pathways -> {len(filtered)} after filtering")
    return filtered


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


def format_label(s, max_len=48, is_tf=False):
    if is_tf:
        s = clean_tf_name(s)
        s = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
    if len(s) <= max_len:
        return s
    # Try to wrap at a word boundary
    cut = s.rfind(" ", 0, max_len)
    if cut > max_len // 2:
        return s[:cut] + "\n" + s[cut + 1:]
    return s[: max_len - 1] + "\u2026"


def make_rose_plot(ax, enrich_df, lrt_df, title, is_tf=False, legend_right=False):
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
    if legend_right:
        leg = ax.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.15, 0.5),
            fontsize=4.5, frameon=False, handlelength=0.7, handleheight=0.7,
            labelspacing=0.25, ncol=1, columnspacing=0.8,
        )
    else:
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


# ── Amplitude computation ──────────────────────────────────────────────────


def compute_amplitude(samples):
    """Compute amplitude (max - min over ZTs) per sample.

    Args:
        samples: np.ndarray [num_genes, num_samples, 4] in log-rate space

    Returns:
        np.ndarray [num_genes, num_samples]
    """
    return samples.max(axis=-1) - samples.min(axis=-1)


# ── Downsampled data loading ──────────────────────────────────────────────


def load_downsampled_metrics(cluster, condition):
    """Load de_novo_metrics from downsampled M/F comparison data.

    Note: This data lacks expected_mesor and expected_acrophase columns.
    """
    path = os.path.join(DOWNSAMPLED_DIR, cluster, condition, "de_novo_metrics.tsv")
    return pd.read_table(path, sep="\t", index_col="gene")


def load_downsampled_waveform_params(cluster, condition):
    """Load alpha, beta, min/max from downsampled data."""
    base = os.path.join(DOWNSAMPLED_DIR, cluster, condition)
    alpha_df = pd.read_table(os.path.join(base, "gene_log_alpha.tsv"), sep="\t", index_col="gene")
    beta_df = pd.read_table(os.path.join(base, "gene_log_beta.tsv"), sep="\t", index_col="gene")
    minmax_df = pd.read_table(os.path.join(base, "log_min_max.tsv"), sep="\t", index_col="gene")
    return alpha_df, beta_df, minmax_df


def filter_cyclers_downsampled(metric_df, bf_threshold=WAVEFORM_BF_MIN):
    """Filter cyclers from downsampled data (no expected_mesor/acrophase).

    Applies: frac_cell_detected, num_cell_detected,
             frac_circadian_largest_component, BF.
    """
    mask = (
        (metric_df["frac_cell_detected"] > FRAC_CELL_DETECTED_MIN)
        & (metric_df["num_cell_detected"] > NUM_CELL_DETECTED_MIN)
        & (metric_df["frac_circadian_samples_largest_component"] >= FRAC_CIRC_LARGEST_COMP_MIN)
        & (metric_df["waveform_over_circadian_component_subtracted_log10_bf"] >= bf_threshold)
    )
    return set(metric_df.index[mask])


# ── SMC switching gene sets ───────────────────────────────────────────────


def load_smc_switching_genes():
    """Load SMC phenotypic switching gene sets (154 up, 233 down).

    Returns:
        dict with keys 'up' and 'down', values are lists of gene names.
    """
    def _read(filename):
        path = os.path.join(SMC_SWITCHING_DIR, filename)
        with open(path) as f:
            genes = [line.strip() for line in f if line.strip()]
        return [g[0].upper() + g[1:] if g else g for g in genes]
    return {"up": _read("up_up.txt"), "down": _read("down_down.txt")}


# ── Posterior waveform plotting ───────────────────────────────────────────

ZT_HOURS = [0, 6, 12, 18]


def plot_posterior_waveform(ax, samples, color, label, ci_level=0.95):
    """Plot posterior waveform mean +/- CI ribbon for a single gene.

    Args:
        samples: np.ndarray [num_samples, 4]
        color: line/fill color
        label: legend label
        ci_level: credible interval level
    """
    mean = samples.mean(axis=0)
    lo = np.quantile(samples, (1 - ci_level) / 2, axis=0)
    hi = np.quantile(samples, 1 - (1 - ci_level) / 2, axis=0)
    ax.plot(ZT_HOURS, mean, color=color, linewidth=1.0, label=label, zorder=3)
    ax.fill_between(ZT_HOURS, lo, hi, color=color, alpha=0.2, zorder=2)


def plot_posterior_violins(ax, samples_dict, colors, width=2.0, alpha=0.6):
    """Plot posterior waveform samples as violins at each ZT.

    Args:
        ax: matplotlib Axes
        samples_dict: dict {label: np.ndarray [num_samples, 4]}
        colors: dict {label: color}
        width: total width budget per ZT point (split across conditions)
        alpha: violin fill alpha
    """
    labels = list(samples_dict.keys())
    n_cond = len(labels)
    half = width / 2
    offsets = np.linspace(-half * (n_cond - 1) / n_cond,
                          half * (n_cond - 1) / n_cond, n_cond)

    for ci, label in enumerate(labels):
        samp = samples_dict[label] / np.log(10)  # convert ln → log10
        color = colors[label]
        positions = [zt + offsets[ci] for zt in ZT_HOURS]
        parts = ax.violinplot([samp[:, t] for t in range(4)],
                              positions=positions,
                              widths=width / n_cond * 0.9,
                              showmeans=False, showmedians=True,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.5)
        parts["cmedians"].set_color(color)
        parts["cmedians"].set_linewidth(0.8)
        # Dummy line for legend
        ax.plot([], [], color=color, linewidth=1.5, label=label)

    ax.set_xticks(ZT_HOURS)
    ax.set_xticklabels(ZT_LABELS, fontsize=5)
