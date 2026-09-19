"""Export posterior waveform parameters for the static gene browser.

For each gene × cell_type × condition, stores the Beta posterior parameters as a
flat list [log_min, log_max, log_alpha x4, log_beta x4] (ZT0, 6, 12, 18). The
browser computes quantiles and violin densities from these in closed form.
Output is a set of chunked JSON files (~100 genes each) plus an index mapping
gene names to chunk filenames.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
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

CHUNK_SIZE = 100


def export_params(alpha_df, beta_df, minmax_df):
    """Return {gene: [log_min, log_max, log_alpha x4, log_beta x4]}."""
    genes = list(alpha_df.index)
    mm = minmax_df.loc[genes, ["log_min", "log_max"]].values
    params = np.hstack([mm, alpha_df.values, beta_df.values]).round(3)
    return {g: params[i].tolist() for i, g in enumerate(genes)}


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

            print(f"  Exporting {ct}/{cond} ({len(alpha)} genes)...", flush=True)
            params = export_params(alpha, beta_df, mm)

            for gene, q in params.items():
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
