"""Create a cleaned, compressed h5ad for public sharing.

Reads the original adata_qc_filtered.h5ad (unchanged), filters to qc_pass cells,
casts X to int32, drops recomputable obs/var columns, and saves with gzip compression.
"""

import anndata as ad
import numpy as np
import scipy.sparse as sp

INPUT = "/Users/mingyaolab/Dropbox/aorta_circadian_data/datasets/joint/adata_qc_filtered.h5ad"
OUTPUT = "/Users/mingyaolab/Dropbox/aorta_circadian_data/datasets/joint/adata_qc_filtered_shared.h5ad"

print("Loading original h5ad (this may take a minute)...", flush=True)
adata = ad.read_h5ad(INPUT)
print(f"  Original: {adata.n_obs} cells, {adata.n_vars} genes")

# Filter to QC-passing cells
if "qc_pass" in adata.obs.columns:
    n_before = adata.n_obs
    adata = adata[adata.obs["qc_pass"]].copy()
    print(f"  After qc_pass filter: {adata.n_obs} cells (removed {n_before - adata.n_obs})")

# Cast X to int32 (UMI counts are integers)
if sp.issparse(adata.X):
    adata.X = adata.X.astype(np.int32)
    print(f"  X cast to int32 sparse ({adata.X.format}), nnz={adata.X.nnz:,}")
else:
    adata.X = adata.X.astype(np.int32)
    print(f"  X cast to int32 dense")

# Drop recomputable obs columns
drop_obs = [
    "log1p_n_genes_by_counts",
    "log1p_total_counts",
    "pct_counts_in_top_50_genes",
    "pct_counts_in_top_100_genes",
    "pct_counts_in_top_200_genes",
    "pct_counts_in_top_500_genes",
    "doublet_score",
    "mito_qc_pass",
    "umi_qc_pass",
    "doublet_qc_pass",
    "qc_pass",
]
existing_drop_obs = [c for c in drop_obs if c in adata.obs.columns]
adata.obs = adata.obs.drop(columns=existing_drop_obs)
print(f"  Dropped {len(existing_drop_obs)} recomputable obs columns")
print(f"  Remaining obs columns: {list(adata.obs.columns)}")

# Drop per-batch var statistics (recomputable)
drop_var = [c for c in adata.var.columns if any(
    c.startswith(p) for p in ["n_cells_by_counts-", "mean_counts-", "log1p_mean_counts-",
                               "pct_dropout_by_counts-", "total_counts-", "log1p_total_counts-"]
)]
adata.var = adata.var.drop(columns=drop_var)
print(f"  Dropped {len(drop_var)} recomputable var columns")
print(f"  Remaining var columns: {list(adata.var.columns)}")

print(f"\nSaving to {OUTPUT} with gzip compression...", flush=True)
adata.write_h5ad(OUTPUT, compression="gzip")

import os
orig_size = os.path.getsize(INPUT) / 1e9
new_size = os.path.getsize(OUTPUT) / 1e9
print(f"\nDone!")
print(f"  Original: {orig_size:.2f} GB")
print(f"  Cleaned:  {new_size:.2f} GB ({new_size/orig_size*100:.0f}% of original)")
