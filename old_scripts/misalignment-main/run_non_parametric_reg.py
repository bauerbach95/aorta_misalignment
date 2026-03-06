import argparse
import sys
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import anndata
import tqdm
import pandas as pd
import anndata
import csv
import gzip
import os
import scipy.io
import anndata
import scanpy
import tempo
import tempo2
from tempo2 import identify_de_novo_cyclers
import torch


# python run_non_parametric_reg.py -f adata_qc_filtered.h5ad -cf clusters.tsv -c 5 -d 'male aligned bmal1-control' -o temp

def main(argv):

	# --- PARSE INPUT ARGUMENTS ---

	# ** init **
	parser = argparse.ArgumentParser()

	# ** .h5ad anndata filepath **
	parser.add_argument("-f", help=".h5ad AnnData file", required=True)

	# ** cluster filepath **
	parser.add_argument("-cf", help="Cluster file", required=True)

	# ** genes to est path **
	parser.add_argument("-gf", help="Genes to estimate file", required=False)

	# ** cluster **
	parser.add_argument("-c", help="Cluster", required=True)

	# ** description **
	parser.add_argument("-d", help="Description", required=True)

	# ** folder out for the results **
	parser.add_argument("-o", help="Folder out", required=True)

	# ** parse **
	args = parser.parse_args()


	# --- LOAD adata ---
	adata = anndata.read_h5ad(args.f)


	# --- LOAD THE CLUSTERS ---

	# ** load clusters **
	cluster_df = pd.read_table(args.cf,sep='\t',index_col='index')

	# ** make sure everything in the same order
	adata = adata[list(cluster_df.index)]

	# ** add clusters **
	try:
		adata.obs["cluster"] = np.array(cluster_df['smc_subcluster'])
	except:
		adata.obs["cluster"] = np.array(cluster_df['leiden_scvi_cluster'])



	# --- LOAD THE GENES TO ESTIMATE IF SUPPLIED ---
	if args.gf is not None:
		with open(args.gf) as file_obj:
			genes_to_est = list(map(lambda x: x.replace("\n",""), file_obj.readlines()))
	else:
		genes_to_est = list(adata.var_names)


	# --- ADD PHASE COL ---

	adata.obs['phase'] = np.array((adata.obs['zt'] / 24.0) * 2 * np.pi)




	# --- RUN REG ---

	# ** get cluster, condition adata **
	cluster_condition_adata = adata[(adata.obs['cluster'] == int(args.c)) & (adata.obs['description'] == args.d)]
	cluster_condition_adata = cluster_condition_adata[:,genes_to_est]
	print(cluster_condition_adata.shape)

	
	# ** prep **
	cluster_condition_adata.var['prop'] = np.array(np.sum(cluster_condition_adata.X,axis=0)).flatten() / np.sum(cluster_condition_adata.obs['lib_size'])
	if 'log_L' not in adata.obs:
		cluster_condition_adata.obs['log_L'] = np.array(np.log(cluster_condition_adata.obs['lib_size']))
	cluster_condition_adata = cluster_condition_adata[:,cluster_condition_adata.var['prop'] > 0]

	
	
	# ** run **
	tempo2.identify_de_novo_cyclers.run(adata = cluster_condition_adata,
		folder_out = args.o,
		num_grid_points = 4,
		phases_sampled = torch.Tensor(np.array(cluster_condition_adata.obs['phase'])).unsqueeze(1), # torch.Tensor(np.array(adata.obs['phase'])).unsqueeze(1),
		cell_phase_dist = None, # cell_posterior_obj, # cell_posterior_obj # cell_posterior_obj
		use_nb = True,
		log_mean_log_disp_coef= torch.Tensor(np.array([-3.001958370208740234e+00, -1.134198158979415894e-01])),# None, torch.Tensor(np.array([-3.001958370208740234e+00, -1.134198158979415894e-01]))
		lr = 1e-1,
		num_waveform_est_cell_samples = 1,
		num_waveform_est_gene_samples = 1,
		num_bf_est_cell_samples = 5,
		num_bf_est_gene_samples = 5,
		vi_max_epochs = 300,
		vi_print_epoch_loss = True,
		vi_improvement_window = 5,
		vi_convergence_criterion = 1e-3,
		cosinor_zero_frac_num_samples_per_waveform = 100,
		num_waveform_bf_cell_samples = 5,
		num_waveform_bf_gene_samples = 5,
		)

			




if __name__ == "__main__":
	main(sys.argv)



