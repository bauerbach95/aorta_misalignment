#!/usr/bin/env python3
"""Create downsampled data for WT vs iKO comparison, per cell type.

Usage:
    python create_downsampled_fibroblast_ko_data.py          # default: cluster 1 (Fibroblast)
    python create_downsampled_fibroblast_ko_data.py --cluster 0  # cluster 0 (SMC)

Replicates the pipeline from assess_if_cell_types_equally_rhythmic.ipynb
but does cluster-specific (not global) downsampling for better power:
1. Load adata + clusters
2. For the target cluster, extract cells for 4 conditions (male/female x WT/KO)
3. Downsample cells per timepoint to the minimum across conditions
4. Downsample UMIs (pseudobulk library size) to match
5. Run tempo2 Bayesian non-parametric regression for each condition
6. Output de_novo_metrics.tsv with frac_cell_detected and num_cell_detected
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import scipy.sparse as sp
import anndata
import pandas as pd
import torch
import argparse
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=int, default=1, help='Cluster index (0=SMC, 1=Fibroblast)')
args = parser.parse_args()


def downsample_counts_total(X, target_total):
    """Downsample total UMI counts in a count matrix to `target_total`.

    Equivalent to scanpy.pp.downsample_counts(total_counts=target_total).
    Uses multinomial sampling: each UMI is independently assigned to a
    (cell, gene) slot with probability proportional to its current count.
    Modifies X in place (dense) or returns new array.
    """
    if sp.issparse(X):
        X_dense = np.array(X.todense())
    else:
        X_dense = np.asarray(X)

    current_total = int(X_dense.sum())
    if current_total <= target_total:
        return

    flat = X_dense.flatten().astype(np.float64)
    probs = flat / flat.sum()
    sampled = np.random.multinomial(target_total, probs)
    result = sampled.reshape(X_dense.shape).astype(X_dense.dtype)

    if sp.issparse(X):
        X_sp = sp.csr_matrix(result)
        X.__dict__.update(X_sp.__dict__)
    else:
        X[:] = result


# ── Paths ─────────────────────────────────────────────────────────────────────

head_folder = "/".join(os.getcwd().split("/")[:3])
data_head_folder = f'{head_folder}/Dropbox/aorta_circadian_data/datasets/joint'
adata_path = f'{data_head_folder}/adata_qc_filtered.h5ad'
clustering_folder = (
    f'{data_head_folder}/data_annotations/hvg/prior_knowledge_guided/'
    f'scvi_res/clustering/res_0.05'
)
cluster_df_path = f'{clustering_folder}/clusters.tsv'

output_head = f'{clustering_folder}/male_cell_type_rhythmicity_same_lib_size_and_cell_count'

CLUSTER = args.cluster
CLUSTER_NAMES = {0: 'SMC', 1: 'Fibroblast', 2: 'EC', 3: 'Macrophage'}
print(f"Processing cluster {CLUSTER} ({CLUSTER_NAMES.get(CLUSTER, 'unknown')})")
CONDITIONS = [
    'male aligned bmal1-control',
    'male aligned bmal1-ko',
    'female aligned bmal1-control',
    'female aligned bmal1-ko',
]
ZTS = [0, 6, 12, 18]
MIN_PROP = 1e-7

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading AnnData...")
adata = anndata.read_h5ad(adata_path)
print(f"  Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")

print("Loading clusters...")
cluster_df = pd.read_table(cluster_df_path, sep='\t', index_col='index')
adata = adata[list(cluster_df.index)]
adata.obs["cluster"] = np.array(cluster_df['leiden_scvi_cluster'])

# ── Filter to cluster_1 and relevant conditions ──────────────────────────────

mask = (adata.obs['cluster'] == CLUSTER) & (adata.obs['description'].isin(CONDITIONS))
adata = adata[mask].copy()
print(f"  Cluster {CLUSTER} with WT/KO conditions: {adata.shape[0]} cells")

# ── Filter genes by pseudobulk threshold ──────────────────────────────────────

adata.var['pseudobulk_count'] = np.array(np.sum(adata.X, axis=0)).flatten()
pseudobulk_threshold = MIN_PROP * np.sum(adata.obs['lib_size'])
adata = adata[:, adata.var['pseudobulk_count'] >= pseudobulk_threshold]
print(f"  After pseudobulk filter: {adata.shape[1]} genes")

# ── Identify genes meeting minimum proportion in each condition ───────────────

genes_to_est = set()
for cond in CONDITIONS:
    cond_adata = adata[adata.obs['description'] == cond]
    cond_adata.var['prop'] = (
        np.array(np.sum(cond_adata.X, axis=0)).flatten()
        / np.sum(cond_adata.obs['lib_size'])
    )
    passing = cond_adata[:, cond_adata.var['prop'] >= MIN_PROP].var_names
    genes_to_est.update(list(passing))
    print(f"  {cond}: {len(passing)} genes pass min_prop")

genes_to_est = sorted(genes_to_est)
print(f"  Union of genes to estimate: {len(genes_to_est)}")

# ── Add phase column ─────────────────────────────────────────────────────────

adata.obs['phase'] = np.array((adata.obs['zt'] / 24.0) * 2 * np.pi)

# ── Compute minimum cells per timepoint ───────────────────────────────────────

print("\nCell counts per condition/timepoint:")
min_num_cells = np.inf
for cond in CONDITIONS:
    for zt in ZTS:
        n = adata[(adata.obs['description'] == cond) & (adata.obs['zt'] == zt)].shape[0]
        min_num_cells = min(min_num_cells, n)
        print(f"  {cond} ZT{zt}: {n}")

min_num_cells = int(min_num_cells)
print(f"\nMinimum cells per timepoint: {min_num_cells}")

# ── Downsample cells ──────────────────────────────────────────────────────────

print("Downsampling cells...")
barcodes_to_keep = []
for cond in CONDITIONS:
    for zt in ZTS:
        sub = adata[(adata.obs['description'] == cond) & (adata.obs['zt'] == zt)]
        sampled = np.random.choice(
            list(sub.obs.index), size=min_num_cells, replace=True
        )
        barcodes_to_keep.extend(sampled)

adata = adata[barcodes_to_keep].copy()
print(f"  After cell downsampling: {adata.shape[0]} cells")

# ── Ensure dense matrix for downsampling ──────────────────────────────────────

if sp.issparse(adata.X):
    adata.X = np.array(adata.X.todense())

# ── Compute minimum pseudobulk library size ───────────────────────────────────

min_tx = np.inf
for cond in CONDITIONS:
    for zt in ZTS:
        sub = adata[(adata.obs['description'] == cond) & (adata.obs['zt'] == zt)]
        pseudobulk_lib = np.sum(sub.X)
        min_tx = min(min_tx, pseudobulk_lib)

min_tx = int(min_tx)
print(f"  Minimum pseudobulk library size: {min_tx}")

# ── Downsample UMIs ───────────────────────────────────────────────────────────

print("Downsampling UMIs...")
for cond in CONDITIONS:
    for zt in ZTS:
        idx = (adata.obs['description'] == cond) & (adata.obs['zt'] == zt)
        row_mask = np.array(idx)
        sub_X = adata.X[row_mask]
        current_total = int(sub_X.sum())
        if current_total > min_tx:
            flat = sub_X.flatten().astype(np.float64)
            probs = flat / flat.sum()
            sampled = np.random.multinomial(min_tx, probs)
            adata.X[row_mask] = sampled.reshape(sub_X.shape).astype(sub_X.dtype)
        print(f"  {cond} ZT{zt}: {current_total} -> {int(adata.X[row_mask].sum())}")

# ── Update lib_size after UMI downsampling ────────────────────────────────────

print("Updating lib_size...")
adata.obs['lib_size'] = np.array(np.sum(adata.X, axis=1)).flatten()

# ── Validate UMI counts ──────────────────────────────────────────────────────

print("\nValidating UMI totals per condition/timepoint:")
for cond in CONDITIONS:
    for zt in ZTS:
        sub = adata[(adata.obs['description'] == cond) & (adata.obs['zt'] == zt)]
        total = np.sum(sub.X)
        print(f"  {cond} ZT{zt}: {total}")

# ── Run tempo2 regression ─────────────────────────────────────────────────────

import tempo2
from tempo2 import identify_de_novo_cyclers

print("\n" + "=" * 70)
print("Running tempo2 Bayesian non-parametric regression")
print("=" * 70)

for cond in CONDITIONS:
    print(f"\n--- Cluster {CLUSTER}, {cond} ---")

    folder_out = os.path.join(output_head, f'cluster_{CLUSTER}', cond)
    os.makedirs(folder_out, exist_ok=True)

    cond_adata = adata[adata.obs['description'] == cond].copy()
    cond_adata = cond_adata[:, genes_to_est].copy()
    X_dense = cond_adata.X.toarray() if sp.issparse(cond_adata.X) else np.asarray(cond_adata.X)
    cond_adata.X = sp.csr_matrix(X_dense)
    print(f"  Cells: {cond_adata.shape[0]}, Genes: {cond_adata.shape[1]}")

    cond_adata.var['prop'] = (
        np.array(np.sum(cond_adata.X, axis=0)).flatten()
        / np.sum(cond_adata.obs['lib_size'])
    )
    if 'log_L' not in cond_adata.obs:
        cond_adata.obs['log_L'] = np.array(np.log(cond_adata.obs['lib_size']))
    cond_adata = cond_adata[:, cond_adata.var['prop'] > 0].copy()
    X_dense = cond_adata.X.toarray() if sp.issparse(cond_adata.X) else np.asarray(cond_adata.X)
    cond_adata.X = sp.csr_matrix(X_dense)
    print(f"  After prop>0 filter: {cond_adata.shape[1]} genes")

    tempo2.identify_de_novo_cyclers.run(
        adata=cond_adata,
        folder_out=folder_out,
        num_grid_points=4,
        phases_sampled=torch.Tensor(
            np.array(cond_adata.obs['phase'])
        ).unsqueeze(1),
        cell_phase_dist=None,
        use_nb=True,
        log_mean_log_disp_coef=torch.Tensor(
            np.array([-3.001958370208740234e+00, -1.134198158979415894e-01])
        ),
        lr=1e-1,
        num_waveform_est_cell_samples=1,
        num_waveform_est_gene_samples=1,
        num_bf_est_cell_samples=5,
        num_bf_est_gene_samples=5,
        vi_max_epochs=300,
        vi_print_epoch_loss=True,
        vi_improvement_window=5,
        vi_convergence_criterion=1e-3,
        cosinor_zero_frac_num_samples_per_waveform=100,
        num_waveform_bf_cell_samples=5,
        num_waveform_bf_gene_samples=5,
    )

    de_novo_path = os.path.join(folder_out, 'de_novo_metrics.tsv')
    de_novo_df = pd.read_table(de_novo_path, sep='\t', index_col='gene')

    cond_adata = cond_adata[:, de_novo_df.index]
    val = np.array(np.sum(cond_adata.X > 0, axis=0)).flatten()
    de_novo_df['num_cell_detected'] = val
    de_novo_df['frac_cell_detected'] = val / cond_adata.shape[0]
    de_novo_df.to_csv(de_novo_path, sep='\t')

    n_cyc = de_novo_df[
        (de_novo_df['waveform_over_circadian_component_subtracted_log10_bf'] >= 2)
        & (de_novo_df['frac_cell_detected'] > 0.01)
    ].shape[0]
    print(f"  Cyclers (BF>=2, frac>0.01): {n_cyc}")

print("\n" + "=" * 70)
print("Done! All conditions processed.")
print(f"Output: {output_head}/cluster_{CLUSTER}/")
print("=" * 70)
