import scanpy as sc
import anndata
import numpy as np
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import os
import pandas as pd
import joblib


# --- PARAMS FOR DE ---

# get the user folder
user_folder = "/".join(os.getcwd().split("/")[:3])

# which data to use
head_folder = '%s/Dropbox/aorta_circadian_data/datasets/joint' % user_folder
hvg_to_use = 'transformed_X_outlier_variance'
cluster_resolution = 0.05


# get corresponding paths
adata_path = '%s/adata_qc_filtered.h5ad' % head_folder
hvg_folder = '%s/data_annotations/hvg/%s' % (head_folder,hvg_to_use)
scvi_res_folder = '%s/scvi_res' % hvg_folder
scvi_mean_embedding_df_path = '%s/scvi_mean_embedding.tsv' % scvi_res_folder
umap_path_out = '%s/scvi_mean_umap_embedding.tsv' % scvi_res_folder
clustering_folder = '%s/clustering/res_%s' % (scvi_res_folder, str(cluster_resolution))
cluster_df_fileout = '%s/clusters.tsv' % (clustering_folder)
reg_head_folder = '%s/bayesian_harmonic_reg' % clustering_folder
torch_model_path_out = '%s/scvi_model_torch.pt' % scvi_res_folder


# num markers per cluster
num_markers_per_cluster = 30

# DE threshold
de_threshold = 0.9


# parameters for calling cell type markers
num_cell_samples = 2000



# --- LOAD ADATA ---

adata = anndata.read_h5ad(adata_path)

adata

# --- LOAD HVG AND LIMIT ADATA TO THEM ---
hv_path = '%s/var_names.csv' % torch_model_path_out
hvg_df = pd.read_table(hv_path,sep=',',header=None)
hvg = list(hvg_df.iloc[:,0])
adata = adata[:,hvg]


# --- MAKE ADD size_factors COLUMN ---
adata.obs['size_factors'] = adata.obs['lib_size']


# --- TURN TO RAW COUNTS ---
adata = remove_sparsity(adata)



# --- ADD EMBEDDINGS TO ADATA ---



# ** load embeddings and clusters **
scvi_mean_embedding_df = pd.read_table(scvi_mean_embedding_df_path,sep='\t',index_col='barcode')
umap_embedding_df = pd.read_table(umap_path_out,sep='\t',index_col='barcode')
cluster_df = pd.read_table(cluster_df_fileout,sep='\t',index_col='index')

# ** make sure everything in the same order
adata = adata[list(scvi_mean_embedding_df.index)]
umap_umap_embedding_df = umap_embedding_df.loc[list(scvi_mean_embedding_df.index)]
cluster_df = cluster_df.loc[list(scvi_mean_embedding_df.index)]


# ** add embeddings **
adata.obsm["X_scVI"] = np.array(scvi_mean_embedding_df)
adata.obsm["X_scVI_umap"] = np.array(umap_embedding_df)
adata.obs["cluster"] = np.array(cluster_df['leiden_scvi_cluster'])


# --- LOAD SCVI VAE ---
model_path_out = '%s/scvi_model.pkl' % scvi_res_folder
vae = joblib.load(model_path_out)


# --- SET UP OBS COLUMNS W/ BATCH INFO, COVARIATES ETC. FOR scANVI TO USE ---

# inputs
categorical_covariate_keys = ['bmal1_ko', 'sex', 'misaligned', 'zt']
batch_key = 'batch'
labels_key = None

# get cols
sca.dataset.setup_anndata(adata,
	batch_key=batch_key,
	labels_key=labels_key,
	categorical_covariate_keys=categorical_covariate_keys)


# --- GET UNIQUE CLUSTERS ---
clusters = sorted(list(adata.obs['cluster'].unique()))


# --- OPTIONAL: JUST SUBSAMPLE CELLS ---


# --- DEFINE FUNCTION TO RETURN FRACTION LARGER FOR TWO PAIRS OF ADATA ---

def bayesian_mean_dif(adata_one, adata_two, num_cell_samples):

	# sample cels
	one_sampled_cell_indices = np.random.choice(adata_one.obs.index,size=num_cell_samples,replace=True)
	two_sampled_cell_indices = np.random.choice(adata_two.obs.index,size=num_cell_samples,replace=True)

	# sample means for one and two cells
	one_means = vae.get_likelihood_parameters(adata_one[one_sampled_cell_indices])['mean']
	two_means = vae.get_likelihood_parameters(adata_two[two_sampled_cell_indices])['mean']

	# compute difference of means and fraction of samples above
	dif_means = one_means - two_means # [num_cell_samples x num_genes]
	
	return dif_means

def fraction_larger_and_smaller_bayesian_de(adata_one, adata_two, num_cell_samples):
	dif_means = bayesian_mean_dif(adata_one, adata_two, num_cell_samples)
	fraction_larger_samples = np.sum((dif_means > 0),axis=0) / num_cell_samples # [num_genes]
	fraction_smaller_samples = np.sum((dif_means < 0),axis=0) / num_cell_samples # [num_genes]
	return fraction_larger_samples, fraction_smaller_samples


# # --- RUN ONE VS. ALL DE ---


# # init the matrix storing the fraction above and below
# fraction_larger_samples_mat = np.zeros((len(clusters),adata.shape[1])) # [num_clusters x num_genes]
# fraction_smaller_samples_mat = np.zeros((len(clusters),adata.shape[1])) # [num_clusters x num_genes]
# gene_samples_mat = np.zeros((len(clusters),num_cell_samples,adata.shape[1])) # [num_clusters x num_genes]

# # fill
# for cluster_index, cluster in enumerate(clusters):

# 	# get cluster of interest and non cluster of interest adata
# 	cluster_adata = adata[adata.obs['cluster'] == cluster]
# 	non_cluster_adata = adata[adata.obs['cluster'] != cluster]

# 	# sample cels
# 	cluster_sampled_cell_indices = np.random.choice(cluster_adata.obs.index,size=num_cell_samples,replace=True)
# 	non_cluster_sampled_cell_indices = np.random.choice(non_cluster_adata.obs.index,size=num_cell_samples,replace=True)

# 	# sample means for cluster and non cluster cells' genes
# 	cluster_means = vae.get_likelihood_parameters(cluster_adata[cluster_sampled_cell_indices])['mean']
# 	non_cluster_means = vae.get_likelihood_parameters(non_cluster_adata[non_cluster_sampled_cell_indices])['mean']

# 	raise Exception("IF GOING TO USE THIS PIECE OF CODE, NEED TO DIVIDE BY LIB SIZES LIKE IN ONE VS. EACH CODE BELOW")


# 	# compute difference of means and fraction of samples above
# 	dif_means = cluster_means - non_cluster_means # [num_cell_samples x num_genes]
# 	fraction_larger_samples_mat[cluster_index,:] = np.sum((dif_means > 0),axis=0) / num_cell_samples # [num_genes]
# 	fraction_smaller_samples_mat[cluster_index,:] = np.sum((dif_means < 0),axis=0) / num_cell_samples # [num_genes]

# 	# add samples to gene_samples_mat
# 	gene_samples_mat[cluster_index,:,:] = cluster_means


# # make fraction_larger_samples_df and fraction_smaller_samples_df
# fraction_larger_samples_df = pd.DataFrame(fraction_larger_samples_mat)
# fraction_larger_samples_df.columns = list(adata.var_names)
# fraction_larger_samples_df['cluster'] = clusters
# fraction_larger_samples_df = fraction_larger_samples_df.set_index('cluster')
# fraction_smaller_samples_df = pd.DataFrame(fraction_smaller_samples_mat)
# fraction_smaller_samples_df.columns = list(adata.var_names)
# fraction_smaller_samples_df['cluster'] = clusters
# fraction_smaller_samples_df = fraction_smaller_samples_df.set_index('cluster')



# --- RUN ONE VS. EACH DE ---

# get unique clusters
clusters = sorted(list(adata.obs['cluster'].unique()))

# init the matrix storing the fraction above and below
each_fraction_larger_samples_mat = np.ones((len(clusters),adata.shape[1])) # [num_clusters x num_genes]
each_fraction_smaller_samples_mat = np.zeros((len(clusters),adata.shape[1])) # [num_clusters x num_genes]
each_gene_samples_mat = np.zeros((len(clusters),num_cell_samples,adata.shape[1])) # [num_clusters x num_genes]

# fill
for cluster_index, cluster in enumerate(clusters):
	temp_each_fraction_larger_samples_mat = np.zeros((len(clusters),adata.shape[1])) # [num_clusters x num_genes]
	temp_each_fraction_smaller_samples_mat = np.zeros((len(clusters),adata.shape[1])) # [num_clusters x num_genes]
	for other_cluster_index, other_cluster in enumerate(clusters):

		print(cluster,other_cluster)

		# handle case when cluster_index == other_cluster_index
		if cluster_index == other_cluster_index:
			temp_each_fraction_larger_samples_mat[cluster_index,:] = np.ones((adata.shape[1]))
			temp_each_fraction_smaller_samples_mat[cluster_index,:] = np.zeros((adata.shape[1]))
			continue

		# get cluster of interest and other of interest adata
		cluster_adata = adata[adata.obs['cluster'] == cluster]
		other_cluster_adata = adata[adata.obs['cluster'] == other_cluster]

		# sample cels
		cluster_sampled_cell_indices = np.random.choice(cluster_adata.obs.index,size=num_cell_samples,replace=True)
		other_cluster_sampled_cell_indices = np.random.choice(other_cluster_adata.obs.index,size=num_cell_samples,replace=True)

		# get the lib sizes for the sampled cells
		cluster_sampled_cell_lib_sizes_latent = vae.get_latent_library_size(cluster_adata[cluster_sampled_cell_indices],give_mean=False)
		other_cluster_sampled_cell_lib_sizes_latent = vae.get_latent_library_size(other_cluster_adata[other_cluster_sampled_cell_indices],give_mean=False)

		# sample means for cluster and non cluster cells' genes
		cluster_means = vae.get_likelihood_parameters(cluster_adata[cluster_sampled_cell_indices])['mean'] / cluster_sampled_cell_lib_sizes_latent
		other_cluster_means = vae.get_likelihood_parameters(other_cluster_adata[other_cluster_sampled_cell_indices])['mean'] / other_cluster_sampled_cell_lib_sizes_latent

		# compute difference of means and fraction of samples above
		dif_means = cluster_means - other_cluster_means # [num_cell_samples x num_genes]
		temp_each_fraction_larger_samples_mat[cluster_index,:] = np.sum((dif_means > 0),axis=0) / num_cell_samples # [num_genes]
		temp_each_fraction_smaller_samples_mat[cluster_index,:] = np.sum((dif_means < 0),axis=0) / num_cell_samples # [num_genes]

	# for each_fraction_larger_samples_mat, make it equal to the minimum of temp_each_fraction_larger_samples_mat across other clusters
	each_fraction_larger_samples_mat[cluster_index,:] = np.min(temp_each_fraction_larger_samples_mat,axis=0)

	# for each_fraction_smaller_samples_mat, make it equal to the maximum of temp_each_fraction_smaller_samples_mat across other clusters	
	each_fraction_smaller_samples_mat[cluster_index,:] = np.max(temp_each_fraction_smaller_samples_mat,axis=0)



# make fraction_larger_samples_df and fraction_smaller_samples_df
each_fraction_larger_samples_df = pd.DataFrame(each_fraction_larger_samples_mat)
each_fraction_larger_samples_df.columns = list(adata.var_names)
each_fraction_larger_samples_df['cluster'] = clusters
each_fraction_larger_samples_df = each_fraction_larger_samples_df.set_index('cluster')
each_fraction_smaller_samples_df = pd.DataFrame(each_fraction_smaller_samples_mat)
each_fraction_smaller_samples_df.columns = list(adata.var_names)
each_fraction_smaller_samples_df['cluster'] = clusters
each_fraction_smaller_samples_df = each_fraction_smaller_samples_df.set_index('cluster')





# # make gene samples df
# gene_samples_df = pd.DataFrame(gene_samples_mat)
# gene_samples_df.columns = list(adata.var_names)
# gene_samples_df['cluster'] = clusters
# gene_samples_df = gene_samples_df.set_index('cluster')


# # --- PICK SOME MARKER GENES FOR EACH CLUSTER AND WRITE THAT OUT FOR THE GENE SAMPLES ---

# cluster_high_markers_list = []
# for cluster_index, cluster in enumerate(clusters):

# 	# check if there are at least num_markers_per_cluster genes that DE at given threshold
# 	cluster_high_markers_sorted = fraction_above_df.loc[cluster].sort_values(ascending=False)
# 	cluster_high_markers = list(cluster_high_markers_sorted[cluster_high_markers_sorted >= de_threshold].index)
# 	cluster_high_markers = clusters_high_markers[:min(len(cluster_high_markers), num_markers_per_cluster)]
# 	cluster_high_markers_list.append(cluster_high_markers)

# # get the cluster high markers
# high_markers = list(np.array(cluster_high_markers_list).flatten())


# # --- RESTRICT GENE SAMPLES MAT TO THE MARKERS ---
# marker_gene_indices = np.where(np.isin(adata.var_names,high_markers))[0]



# --- WRITE RESULTS OUT ---

de_folder_out = '%s/de' % clustering_folder
if not os.path.exists(de_folder_out):
	os.makedirs(de_folder_out)

# # write larger file out
# fileout = '%s/fraction_larger_samples_mat.tsv' % de_folder_out
# fraction_larger_samples_df.to_csv(fileout,sep='\t')

# # write smaller file out
# fileout = '%s/fraction_smaller_samples_mat.tsv' % de_folder_out
# fraction_smaller_samples_df.to_csv(fileout,sep='\t')

# # write gene_samples_mat out
# for cluster_index,cluster in enumerate(clusters):
# 	fileout = '%s/cluster_%s_gene_samples_mat.npy' % (de_folder_out,cluster_index)
# 	# gene_samples_df.to_csv(fileout,sep='\t')
# 	np.savetxt(fileout,gene_samples_mat[cluster_index,:,:])

# write each larger file out
fileout = '%s/each_fraction_larger_samples_mat.tsv' % de_folder_out
each_fraction_larger_samples_df.to_csv(fileout,sep='\t')

# write smaller file out
fileout = '%s/each_fraction_smaller_samples_mat.tsv' % de_folder_out
each_fraction_smaller_samples_df.to_csv(fileout,sep='\t')

