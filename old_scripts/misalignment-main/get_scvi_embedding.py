import scanpy as sc
import anndata
import numpy as np
import scarches as sca
from scarches.dataset.trvae.data_handling import remove_sparsity
import os

# --- INIT ---
head_folder = '/Users/benauerbach/Dropbox/aorta_circadian_data/datasets/joint'
adata_path = '%s/adata_qc_filtered.h5ad' % head_folder
hvg_folder = '%s/data_annotations/hvg/prior_knowledge_guided' % head_folder # transformed_X_outlier_variance
# scvi_res_folder = '%s/scvi_res' % hvg_folder
scvi_res_folder = '%s/scvi_res_2' % hvg_folder
if not os.path.exists(scvi_res_folder):
	os.makedirs(scvi_res_folder)


# --- GET THE PATH OUTS ---
model_path_out = '%s/scvi_model.pkl' % scvi_res_folder
torch_model_path_out = '%s/scvi_model_torch.pt' % scvi_res_folder
adata_path_out = '%s/adata_scvi.h5ad' % scvi_res_folder
scvi_mean_embedding_df_fileout = '%s/scvi_mean_embedding.tsv' % scvi_res_folder



# --- LOAD HVG ---
with open('%s/hvg.txt' % hvg_folder) as file_obj:
	hv_genes = list(map(lambda x: x.replace("\n",""),file_obj.readlines()))



# --- LOAD ---
adata = anndata.read_h5ad(adata_path)


# # --- TEMPORARY ---
# adata = adata[np.arange(0,100)]




# --- LIMIT ADATA TO THE HV GENES ---
hv_genes = list(filter(lambda x: x in adata.var_names,hv_genes))
adata = adata[:,hv_genes]


# --- MAKE ADD size_factors COLUMN ---
adata.obs['size_factors'] = adata.obs['lib_size']



# --- TURN TO RAW COUNTS ---
adata = remove_sparsity(adata)


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


# --- TRAIN VAE ---
vae = sca.models.SCVI(
	adata,
	n_layers=2,
	encode_covariates=True,
	deeply_inject_covariates=True, # False
	use_layer_norm="both",
	use_batch_norm="none",
	gene_likelihood='nb'
)
vae.train()




# --- SAVE MODEL ---
import joblib

joblib.dump(vae,model_path_out)
vae.save(torch_model_path_out)


# # --- ADD INFO ADATA ---
# adata.obsm["X_scVI"] = vae.get_latent_representation()
# adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()


# # --- WRITE ADATA OUT ---
# adata.uns.pop('_scvi',None)
# adata.write(adata_path_out)



# --- WRITE THE SCVI MEAN EMBEDDING OUT DIRECTLY ---
import pandas as pd
scvi_mean_embedding_df = pd.DataFrame()
scvi_mean_embedding_df['barcode'] = np.array(adata.obs.index)
for dim in range(0,vae.get_latent_representation().shape[1]):
	scvi_mean_embedding_df['X_scVI_dim_%s' % dim] = vae.get_latent_representation()[:,dim]
scvi_mean_embedding_df = scvi_mean_embedding_df.set_index("barcode")
scvi_mean_embedding_df.to_csv(scvi_mean_embedding_df_fileout,sep='\t')




# -----------------
# -----------------
# -----------------
# -----------------
# -----------------


















