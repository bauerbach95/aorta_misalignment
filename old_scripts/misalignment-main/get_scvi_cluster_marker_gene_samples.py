
import scanpy as sc
import anndata
import numpy as np
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import os
import pandas as pd
import joblib






# --- INIT ---

# # marker genes for each cell type
# marker_gene_dict = {
# 	0 : ['Ppp1r14a', 'Acta2'],
# 	1: ['Mmp3', 'Dnm1', 'Fbln1'],
# 	2: ['Nos3'],
# 	3: ['Dlx6os1', 'Pianp'],
# 	4: ['Stap1', 'Ms4a6c'],
# 	5: ['Rpl9', 'Adipoq', 'Rpl39', 'Selenoh'],
# 	6: ['Tspan15', 'Cnp', 'Iqgap2', 'Gas2l3'],
# 	7: ['Il7r', 'Gimap3', 'Neurl1b']
# }


# --- PARAMS FOR DE ---

# get the user folder
user_folder = "/".join(os.getcwd().split("/")[:3])

# which data to use
head_folder = '%s/Dropbox/aorta_circadian_data/datasets/joint' % user_folder
hvg_to_use = 'connected'
cluster_resolution = 0.1


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
de_folder_out = '%s/de' % clustering_folder
marker_gene_dict_path = '%s/marker_gene_dict.txt' % (de_folder_out)


# parameters for calling cell type markers
num_cell_samples = 1000


# --- LOAD MARKER GENE DICT ---
with open(marker_gene_dict_path) as file_obj:
	marker_gene_dict = eval(file_obj.read())


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


# --- TOSS AN ERROR IF clusters and keys in marker_gene_dict ARE NOT THE SAME ---

if set(clusters) != set(list(marker_gene_dict.keys())):
	raise Exception("clusters file and clusters in marker gene dict are different.")


# --- GET THE INDICES OF THE MARKER GENES ---

# get a flat list of the marker genes
marker_genes = []
for key,val in marker_gene_dict.items():
	marker_genes += val

# get the indices of the marker genes in the adata
marker_gene_indices = []
for marker_gene in marker_genes:
	marker_gene_indices.append(np.where(adata.var_names == marker_gene)[0][0])




# --- GENERATE GENE SAMPLES FOR EACH CLUSTER / MARKER GENE ---

# init the mat
gene_samples_mat = np.zeros((len(clusters),num_cell_samples,adata.shape[1])) # [num_clusters x num_cell_samples x num_genes]

# fill
for cluster_index, cluster in enumerate(clusters):

	# get cluster of interest and non cluster of interest adata
	cluster_adata = adata[adata.obs['cluster'] == cluster]

	# sample cells
	cluster_sampled_cell_indices = np.random.choice(cluster_adata.obs.index,size=num_cell_samples,replace=True)
	cluster_sampled_cell_lib_sizes = np.array(cluster_adata.obs['lib_size'][cluster_sampled_cell_indices]).reshape(-1,1)
	cluster_sampled_cell_lib_sizes_latent = vae.get_latent_library_size(cluster_adata[cluster_sampled_cell_indices],give_mean=False)

	# sample means for cluster and non cluster cells' genes
	# gene_samples_mat[cluster_index,:,:] = vae.get_likelihood_parameters(cluster_adata[cluster_sampled_cell_indices])['mean']
	# gene_samples_mat[cluster_index,:,:] = vae.get_likelihood_parameters(cluster_adata[cluster_sampled_cell_indices])['mean'] / cluster_sampled_cell_lib_sizes
	gene_samples_mat[cluster_index,:,:] = vae.get_likelihood_parameters(cluster_adata[cluster_sampled_cell_indices])['mean'] / cluster_sampled_cell_lib_sizes_latent


# limit the gene samples mat to the marker genes
gene_samples_mat = gene_samples_mat[:,:,marker_gene_indices]


# --- WRITE OUT ---

de_folder_out = '%s/de' % clustering_folder
if not os.path.exists(de_folder_out):
	os.makedirs(de_folder_out)

# write gene_samples_mat out
for cluster_index,cluster in enumerate(clusters):
	fileout = '%s/cluster_%s_marker_gene_samples_mat.npy' % (de_folder_out,cluster_index)
	# gene_samples_df.to_csv(fileout,sep='\t')
	np.savetxt(fileout,gene_samples_mat[cluster_index,:,:])







