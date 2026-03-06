"""
Generate Figure 2: Cell-type-specific circadian rhythms in mouse aortic cell types.

Subplots:
  a) Bar chart: number of daily cyclers per major cell type (shared M/F)
  b) Rose plots: KEGG pathway acrophase enrichment (SMC, Fibroblast only)
  c) Rose plots: TF acrophase enrichment (SMC, Fibroblast only)

Output:
  figures/fig_2.pdf
  figures/fig_2_source_data/  (raw data tables for Nature submission)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import circmean
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/Users/mingyaolab/Dropbox/aorta_circadian_data"
NONPARAM_REG_DIR = os.path.join(
    DATA_ROOT,
    "datasets/joint/data_annotations/hvg/prior_knowledge_guided/"
    "scvi_res/clustering/res_0.05/nonparametric_reg",
)
GENE_SETS_DIR = os.path.join(DATA_ROOT, "gene_sets")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
SOURCE_DATA_DIR = os.path.join(FIGURES_DIR, "fig_2_source_data")

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

# Cycler thresholds
FRAC_CELL_DETECTED_MIN = 0.01
NUM_CELL_DETECTED_MIN = 50
EXPECTED_MESOR_MIN = -13.8155
FRAC_CIRC_LARGEST_COMP_MIN = 0.6
WAVEFORM_BF_MIN = 2.0
ACROPHASE_HOUR_THRESH = 3.0  # hours

# Enrichment
MIN_GENES_IN_PATHWAY = 6
NUM_ACROPHASE_BINS = 12  # 2-hour bins
LRT_THRESHOLD = 2.0
TOP_N_PATHWAYS = 10

# Cell types with enough cyclers for rose plots
ROSE_PLOT_CELL_TYPES = ["SMC", "Fibroblast"]

# KEGG pathways to exclude (disease/cancer/infection terms)
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

CELL_TYPE_COLORS = {
    "SMC": "#E64B35",
    "Fibroblast": "#4DBBD5",
    "EC": "#00A087",
    "Macrophage": "#3C5488",
}

# ── Matplotlib styling ───────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.weight": "regular",
    "mathtext.default": "regular",
    "pdf.fonttype": 42,  # editable text in PDF
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
})


# ── Helper functions ─────────────────────────────────────────────────────────


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


def acrophase_rad_to_hours(rad):
    return (rad / (2 * np.pi)) * 24.0


def circular_hour_distance(h1, h2, period=24.0):
    d = abs(h1 - h2) % period
    return np.minimum(d, period - d)


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


def compute_acrophase_enrichment(gene_acrophases_rad, gene_set_dict, num_bins=NUM_ACROPHASE_BINS):
    """
    For each gene set, bin member genes' acrophases and compute density/uniform LRT.
    Returns enrichment summary and per-bin LRT values.
    """
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
        # Circular std (concentration) — mean resultant length
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


# ── Step 1: Identify daily cyclers (shared M/F) ─────────────────────────────

print("Step 1: Identifying daily cyclers shared between male and female...")

daily_cyclers = {}
daily_cycler_acrophases = {}
cycler_details = {}  # for source data export

for cluster, cell_type in CLUSTER_CELL_TYPE.items():
    male_df = load_metrics(cluster, ALIGNED_CONDITIONS["male"])
    female_df = load_metrics(cluster, ALIGNED_CONDITIONS["female"])

    male_cyclers = filter_cyclers(male_df)
    female_cyclers = filter_cyclers(female_df)

    shared = male_cyclers & female_cyclers
    agreed = set()
    acrophases = {}
    details_rows = []
    for gene in sorted(shared):
        m_acro = male_df.loc[gene, "expected_acrophase"]
        f_acro = female_df.loc[gene, "expected_acrophase"]
        m_h = acrophase_rad_to_hours(m_acro)
        f_h = acrophase_rad_to_hours(f_acro)
        dist = circular_hour_distance(m_h, f_h)
        if dist <= ACROPHASE_HOUR_THRESH:
            agreed.add(gene)
            avg_acro = circmean([m_acro, f_acro], high=2 * np.pi, low=0)
            acrophases[gene] = avg_acro
            details_rows.append({
                "gene": gene,
                "male_acrophase_hour": round(m_h, 2),
                "female_acrophase_hour": round(f_h, 2),
                "circular_distance_hour": round(float(dist), 2),
                "average_acrophase_hour": round(acrophase_rad_to_hours(avg_acro), 2),
                "average_acrophase_rad": round(float(avg_acro), 4),
                "male_bf_log10": round(male_df.loc[gene, "waveform_over_circadian_component_subtracted_log10_bf"], 2),
                "female_bf_log10": round(female_df.loc[gene, "waveform_over_circadian_component_subtracted_log10_bf"], 2),
            })

    daily_cyclers[cell_type] = agreed
    daily_cycler_acrophases[cell_type] = acrophases
    cycler_details[cell_type] = pd.DataFrame(details_rows)
    print(f"  {cell_type}: {len(agreed)} daily cyclers")


# ── Step 2: KEGG pathway acrophase enrichment ────────────────────────────────

print("\nStep 2: KEGG pathway acrophase enrichment...")

kegg_dict_raw = parse_gsea_set_file(os.path.join(GENE_SETS_DIR, "KEGG_2019_Mouse.txt"))
kegg_dict = filter_kegg_pathways(kegg_dict_raw)
print(f"  {len(kegg_dict_raw)} total KEGG pathways -> {len(kegg_dict)} after filtering")

kegg_enrichments = {}
kegg_enrichments_full = {}  # all enriched (not just top N), for source data
for cell_type in CLUSTER_CELL_TYPE.values():
    enrich_df, lrt_df = compute_acrophase_enrichment(
        daily_cycler_acrophases[cell_type], kegg_dict
    )
    if not enrich_df.empty:
        enrich_df = enrich_df[enrich_df["max_lrt"] >= LRT_THRESHOLD]
    kegg_enrichments_full[cell_type] = (enrich_df.copy(), lrt_df)
    kegg_enrichments[cell_type] = (enrich_df.head(TOP_N_PATHWAYS), lrt_df)
    n = len(enrich_df) if not enrich_df.empty else 0
    print(f"  {cell_type}: {n} enriched pathways (showing top {min(n, TOP_N_PATHWAYS)})")


# ── Step 3: TF acrophase enrichment ──────────────────────────────────────────

print("\nStep 3: TF acrophase enrichment...")

tf_dict = parse_gsea_set_file(os.path.join(GENE_SETS_DIR, "TRANSFAC_and_JASPAR_PWMs.txt"))

tf_enrichments = {}
tf_enrichments_full = {}
for cell_type in CLUSTER_CELL_TYPE.values():
    enrich_df, lrt_df = compute_acrophase_enrichment(
        daily_cycler_acrophases[cell_type], tf_dict
    )
    if not enrich_df.empty:
        enrich_df = enrich_df[enrich_df["max_lrt"] >= LRT_THRESHOLD]
    tf_enrichments_full[cell_type] = (enrich_df.copy(), lrt_df)
    tf_enrichments[cell_type] = (enrich_df.head(TOP_N_PATHWAYS), lrt_df)
    n = len(enrich_df) if not enrich_df.empty else 0
    print(f"  {cell_type}: {n} enriched TFs (showing top {min(n, TOP_N_PATHWAYS)})")


# ── Step 4: Export source data ───────────────────────────────────────────────

print("\nStep 4: Exporting source data...")

os.makedirs(SOURCE_DATA_DIR, exist_ok=True)

# Panel a: bar chart counts
bar_data = pd.DataFrame({
    "cell_type": list(CLUSTER_CELL_TYPE.values()),
    "num_daily_cyclers": [len(daily_cyclers[ct]) for ct in CLUSTER_CELL_TYPE.values()],
})
bar_data.to_csv(os.path.join(SOURCE_DATA_DIR, "panel_a_daily_cycler_counts.csv"), index=False)

# Per-cell-type cycler gene lists
for ct in CLUSTER_CELL_TYPE.values():
    if not cycler_details[ct].empty:
        cycler_details[ct].to_csv(
            os.path.join(SOURCE_DATA_DIR, f"daily_cyclers_{ct}.csv"), index=False
        )

# KEGG enrichment tables (all enriched, not just top N)
for ct in CLUSTER_CELL_TYPE.values():
    enrich_df, lrt_df = kegg_enrichments_full[ct]
    if not enrich_df.empty:
        out = enrich_df.reset_index().rename(columns={"gene_set": "pathway"})
        out.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_b_kegg_enrichment_{ct}.csv"), index=False)
        lrt_df.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_b_kegg_lrt_bins_{ct}.csv"))

# TF enrichment tables
for ct in CLUSTER_CELL_TYPE.values():
    enrich_df, lrt_df = tf_enrichments_full[ct]
    if not enrich_df.empty:
        out = enrich_df.reset_index().rename(columns={"gene_set": "tf_motif"})
        out.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_c_tf_enrichment_{ct}.csv"), index=False)
        lrt_df.to_csv(os.path.join(SOURCE_DATA_DIR, f"panel_c_tf_lrt_bins_{ct}.csv"))

print(f"  Source data written to {SOURCE_DATA_DIR}/")


# ── Step 5: Generate Figure 2 ───────────────────────────────────────────────

print("\nStep 5: Generating figure...")

all_cell_types = ["SMC", "Fibroblast", "EC", "Macrophage"]
NUM_BINS = NUM_ACROPHASE_BINS
BIN_WIDTH = 2 * np.pi / NUM_BINS


def make_rose_plot(ax, enrich_df, lrt_df, title, is_tf=False):
    """Rose plot with subdivided wedge bars for pathways/TFs."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # ZT labels
    zt_ticks = np.linspace(0, 2 * np.pi, NUM_BINS, endpoint=False)
    zt_labels = [f"ZT{h}" for h in range(0, 24, 2)]
    ax.set_xticks(zt_ticks)
    ax.set_xticklabels(zt_labels, fontsize=5.5, fontweight="medium")
    ax.tick_params(axis="y", labelsize=4.5, pad=1)
    ax.tick_params(axis="x", pad=3)

    # Style the polar grid
    ax.grid(True, alpha=0.2, linewidth=0.3, color="0.5")
    ax.spines["polar"].set_linewidth(0.4)
    ax.spines["polar"].set_color("0.7")

    ax.set_title(title, fontsize=8, fontweight="semibold", pad=16,
                 color="0.15")

    if enrich_df.empty:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        return

    pathways = list(enrich_df.index)
    n_pw = len(pathways)

    # Use a nicer colormap
    if n_pw <= 8:
        colors = plt.cm.Set2(np.linspace(0, 1, 8))[:n_pw]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_pw]

    # Group pathways by peak bin
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
            ax.bar(
                theta_center,
                max_lrt,
                width=sub_width * 0.9,
                bottom=0,
                color=colors[i],
                alpha=0.88,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

    # Reference circle at LRT=1 (uniform baseline)
    ref_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ref_theta, np.ones_like(ref_theta), color="0.55",
            linewidth=0.5, linestyle="--", alpha=0.6, zorder=2)

    ax.set_ylim(0, max(max_lrt_val * 1.18, 2.5))

    # Legend — placed below the plot to avoid overlap
    labels = [_format_label(pw, is_tf=is_tf) for pw in pathways]
    handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor="0.85", linewidth=0.3,
                       label=labels[i])
        for i in range(n_pw)
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=4.5,
        frameon=False,
        handlelength=0.7,
        handleheight=0.7,
        labelspacing=0.25,
        ncol=2 if n_pw > 5 else 1,
        columnspacing=0.8,
    )
    for text in leg.get_texts():
        text.set_color("0.2")


def _format_label(s, max_len=32, is_tf=False):
    if is_tf:
        s = clean_tf_name(s)
        s = s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "\u2026"


# ── Figure layout ────────────────────────────────────────────────────────────
# Row a: bar chart (spans left half)
# Row b: 2 KEGG rose plots (SMC, Fibroblast)
# Row c: 2 TF rose plots (SMC, Fibroblast)

n_rose = len(ROSE_PLOT_CELL_TYPES)

fig = plt.figure(figsize=(8.5, 11), dpi=300)
fig.patch.set_facecolor("white")

gs = GridSpec(
    3, n_rose,
    figure=fig,
    hspace=0.75,
    wspace=0.65,
    height_ratios=[0.5, 1.2, 1.2],
    left=0.08, right=0.92,
    top=0.95, bottom=0.04,
)

# ── Row a: Bar chart ─────────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[0, :])

x = np.arange(len(all_cell_types))
daily_counts = [len(daily_cyclers[ct]) for ct in all_cell_types]
bar_colors = [CELL_TYPE_COLORS[ct] for ct in all_cell_types]

bars = ax_bar.bar(x, daily_counts, width=0.55, color=bar_colors,
                  edgecolor="white", linewidth=0.8, zorder=3)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(all_cell_types, fontsize=7, rotation=0, ha="center")
ax_bar.set_ylabel("Daily cyclers\n(shared M & F)", fontsize=7)
ax_bar.set_xlim(-0.6, len(all_cell_types) - 0.4)
ax_bar.tick_params(axis="y", labelsize=6)
sns.despine(ax=ax_bar, bottom=True)
ax_bar.tick_params(axis="x", length=0)
ax_bar.set_axisbelow(True)
ax_bar.yaxis.grid(True, alpha=0.15, linewidth=0.4)

# Count labels
for bar in bars:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h + max(daily_counts) * 0.02,
                str(int(h)), ha="center", va="bottom", fontsize=6.5, fontweight="medium",
                color="0.25")

ax_bar.text(-0.06, 1.08, "a", transform=ax_bar.transAxes,
            fontsize=11, fontweight="bold", va="top")

# ── Row b: KEGG rose plots ──────────────────────────────────────────────────
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
    ax = fig.add_subplot(gs[1, i], projection="polar")
    enrich_df, lrt_df = kegg_enrichments[ct]
    make_rose_plot(ax, enrich_df, lrt_df, ct, is_tf=False)
    if i == 0:
        ax.text(-0.25, 1.22, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Row b label
fig.text(0.02, 0.62, "KEGG Pathways", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# ── Row c: TF rose plots ────────────────────────────────────────────────────
for i, ct in enumerate(ROSE_PLOT_CELL_TYPES):
    ax = fig.add_subplot(gs[2, i], projection="polar")
    enrich_df, lrt_df = tf_enrichments[ct]
    make_rose_plot(ax, enrich_df, lrt_df, ct, is_tf=True)
    if i == 0:
        ax.text(-0.25, 1.22, "c", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

# Row c label
fig.text(0.02, 0.24, "TF Motifs", rotation=90, fontsize=8,
         fontweight="semibold", va="center", color="0.3")

# Save
os.makedirs(FIGURES_DIR, exist_ok=True)
outpath = os.path.join(FIGURES_DIR, "fig_2.pdf")
fig.savefig(outpath, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nFigure saved to {outpath}")
plt.close(fig)
