"""Pre-compute posterior waveform summary statistics for the static gene browser.

For each gene × cell_type × condition, computes the 5th, 50th, and 95th
percentiles of the Beta posterior at each of the 4 ZT bins using the closed-form
inverse CDF (no sampling needed). Output is a set of chunked JSON files (~100
genes each) plus an index mapping gene names to chunk filenames.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from collections import defaultdict

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "bayesian_temporal_regression",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "data",
)

CELL_TYPES = ["SMC", "Fibroblast", "EC", "Macrophage", "SMC0", "SMC1", "SMC2"]
CONDITIONS = [
    "male_aligned_WT",
    "female_aligned_WT",
    "male_aligned_KO",
    "female_aligned_KO",
    "male_misaligned_WT",
    "female_misaligned_WT",
]

CONDITION_LABELS = {
    "male_aligned_WT": "Male Aligned WT",
    "female_aligned_WT": "Female Aligned WT",
    "male_aligned_KO": "Male Aligned iKO",
    "female_aligned_KO": "Female Aligned iKO",
    "male_misaligned_WT": "Male Misaligned WT",
    "female_misaligned_WT": "Female Misaligned WT",
}

QUANTILES = [0.05, 0.50, 0.95]
CHUNK_SIZE = 100


def compute_quantiles(alpha_df, beta_df, minmax_df):
    """Compute exact Beta quantiles for all genes (closed-form, no sampling).

    Returns dict: {gene: [[p05, p50, p95] for each of 4 ZT bins]}
    """
    genes = list(alpha_df.index)
    alpha = np.exp(alpha_df.values)   # [num_genes, 4]
    beta_vals = np.exp(beta_df.values)  # [num_genes, 4]
    log_min = minmax_df.loc[genes, "log_min"].values[:, np.newaxis]  # [num_genes, 1]
    log_max = minmax_df.loc[genes, "log_max"].values[:, np.newaxis]

    result = {}
    for qi, q in enumerate(QUANTILES):
        raw = beta_dist.ppf(q, alpha, beta_vals)  # [num_genes, 4] in [0,1]
        scaled = raw * (log_max - log_min) + log_min
        for i, gene in enumerate(genes):
            if gene not in result:
                result[gene] = [[] for _ in range(4)]
            for zt in range(4):
                result[gene][zt].append(round(float(scaled[i, zt]), 3))

    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_data = defaultdict(lambda: defaultdict(dict))
    all_genes = set()

    for ct in CELL_TYPES:
        for cond in CONDITIONS:
            path = os.path.join(DATA_DIR, ct, cond)
            if not os.path.isdir(path):
                print(f"  SKIP {ct}/{cond} (not found)")
                continue

            alpha = pd.read_csv(os.path.join(path, "gene_log_alpha.tsv"), sep="\t", index_col="gene")
            beta_df = pd.read_csv(os.path.join(path, "gene_log_beta.tsv"), sep="\t", index_col="gene")
            mm = pd.read_csv(os.path.join(path, "log_min_max.tsv"), sep="\t", index_col="gene")

            print(f"  Computing {ct}/{cond} ({len(alpha)} genes)...", flush=True)
            quantiles = compute_quantiles(alpha, beta_df, mm)

            for gene, q in quantiles.items():
                all_data[gene][ct][cond] = q
                all_genes.add(gene)

    print(f"\nTotal unique genes: {len(all_genes)}")

    sorted_genes = sorted(all_genes)
    chunks = []
    gene_to_chunk = {}
    for i in range(0, len(sorted_genes), CHUNK_SIZE):
        chunk_genes = sorted_genes[i : i + CHUNK_SIZE]
        chunk_name = f"chunk_{i // CHUNK_SIZE:04d}.json"
        chunk_data = {}
        for g in chunk_genes:
            chunk_data[g] = dict(all_data[g])
            gene_to_chunk[g] = chunk_name
        chunks.append((chunk_name, chunk_data))

    print(f"Writing {len(chunks)} chunk files...")
    for chunk_name, chunk_data in chunks:
        with open(os.path.join(OUT_DIR, chunk_name), "w") as f:
            json.dump(chunk_data, f, separators=(",", ":"))

    index = {
        "genes": sorted_genes,
        "gene_to_chunk": gene_to_chunk,
        "cell_types": CELL_TYPES,
        "conditions": CONDITIONS,
        "condition_labels": CONDITION_LABELS,
        "zt_hours": [0, 6, 12, 18],
    }
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(index, f, separators=(",", ":"))

    total_size = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR)
        if f.endswith(".json")
    )
    print(f"Total data size: {total_size / 1e6:.1f} MB across {len(chunks) + 1} files")
    print("Done!")


if __name__ == "__main__":
    main()
