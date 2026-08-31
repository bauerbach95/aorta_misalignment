"""Pre-compute posterior waveform summary statistics for the static gene browser.

For each gene × cell_type × condition, samples 2000 posterior waveforms and
stores the 5th, 50th, and 95th percentiles at each of the 4 ZT bins. Output is
a set of chunked JSON files (~100 genes each) plus an index mapping gene names
to chunk filenames.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "bayesian_temporal_regression",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "browser", "data",
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

NUM_SAMPLES = 2000
CHUNK_SIZE = 100


def sample_quantiles(alpha_df, beta_df, minmax_df, num_samples=NUM_SAMPLES):
    """Sample waveforms for ALL genes and return quantiles.

    Returns dict: {gene: [[p05, p50, p95] for each of 4 ZT bins]}
    """
    genes = list(alpha_df.index)
    alpha = np.exp(alpha_df.values)
    beta = np.exp(beta_df.values)
    log_min = minmax_df.loc[genes, "log_min"].values
    log_max = minmax_df.loc[genes, "log_max"].values

    dist = torch.distributions.Beta(
        torch.tensor(alpha.T, dtype=torch.float32),
        torch.tensor(beta.T, dtype=torch.float32),
    )
    raw = dist.sample((num_samples,)).numpy()  # [num_samples, 4, num_genes]

    log_min_bc = log_min[np.newaxis, np.newaxis, :]
    log_max_bc = log_max[np.newaxis, np.newaxis, :]
    scaled = raw * (log_max_bc - log_min_bc) + log_min_bc  # [num_samples, 4, num_genes]

    p05 = np.percentile(scaled, 5, axis=0)    # [4, num_genes]
    p50 = np.percentile(scaled, 50, axis=0)
    p95 = np.percentile(scaled, 95, axis=0)

    result = {}
    for i, gene in enumerate(genes):
        result[gene] = [
            [round(float(p05[zt, i]), 3), round(float(p50[zt, i]), 3), round(float(p95[zt, i]), 3)]
            for zt in range(4)
        ]
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
            beta = pd.read_csv(os.path.join(path, "gene_log_beta.tsv"), sep="\t", index_col="gene")
            mm = pd.read_csv(os.path.join(path, "log_min_max.tsv"), sep="\t", index_col="gene")

            print(f"  Sampling {ct}/{cond} ({len(alpha)} genes)...")
            quantiles = sample_quantiles(alpha, beta, mm)

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
