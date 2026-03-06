import sys
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import anndata
import tqdm
import pandas as pd



# python concat_temporal_regression_chunks.py -f adata_qc_filtered.h5ad -cf clusters.tsv -o high_level_smc_chunk



# bsub -J concat -e concat.e -o concat.o -M 64000 -R "rusage[mem=64000]" 'source /home/benauer/anaconda2/bin/activate tempo && python concat_temporal_regression_chunks.py -f adata_qc_filtered.h5ad -cf clusters.tsv -o high_level_smc_chunk'


def main(argv):

	# --- PARSE INPUT ARGUMENTS ---

	# ** init **
	parser = argparse.ArgumentParser()

	# ** .h5ad anndata filepath **
	parser.add_argument("-f", help=".h5ad AnnData file", required=True)

	# ** cluster filepath **
	parser.add_argument("-cf", help="Cluster file", required=True)

	# ** folder out for the results **
	parser.add_argument("-o", help="Folder out", required=True)

	# ** parse **
	args = parser.parse_args()





	# --- INIT ---

	adata_path = args.f
	cluster_path = args.cf
	reg_head_folder = args.o
	if not os.path.exists(reg_head_folder):
		os.makedirs(reg_head_folder)
	min_prop = 1e-7
	num_genes_per_chunk = 2000



	# In[5]:


	# --- LOAD ADATA ---

	adata = anndata.read_h5ad(adata_path)

	adata


	# In[6]:


	# --- ADD EMBEDDINGS TO ADATA ---



	# ** load clusters **
	cluster_df = pd.read_table(cluster_path,sep='\t',index_col='index')

	# ** make sure everything in the same order
	adata = adata[list(cluster_df.index)]


	# ** add embeddings **
	try:
		adata.obs["cluster"] = np.array(cluster_df['smc_subcluster'])
	except:
		adata.obs["cluster"] = np.array(cluster_df['leiden_scvi_cluster'])

	adata


	# In[32]:


	# --- GET THE UNIQUE DESCRIPTION AND CLUSTERS ---

	descriptions = list(adata.obs['description'].unique())
	clusters = sorted(list(adata.obs['cluster'].unique()))
	clusters = list(filter(lambda x: ~np.isnan(x),clusters)) # get rid of NaN cluster (only relevant for SMC subcluster)
	clusters = list(map(lambda x: int(x),clusters))



	print("Unique descriptions:\n",descriptions)
	print("Unique clusters:\n",clusters)


	# --- LIMIT ADATA TO GENES THAT MEET MINIMUM PSEUDOBULK COUNT THRESHOLD ---


	# ** get gene pseudobulk counts
	adata.var['pseudobulk_count'] = np.array(np.sum(adata.X,axis=0)).flatten()

	# ** get the number of cells in the smallest cell type
	smallest_cell_type_adata = adata[adata.obs['cluster'] == np.max(clusters)]

	# ** get pseudobulk cutoff **
	pseudobulk_threshold = min_prop * np.sum(smallest_cell_type_adata.obs['lib_size'])

	# ** limit adata to this **
	adata = adata[:,adata.var['pseudobulk_count'] >= pseudobulk_threshold]


	adata


	# In[9]:




	# --- LOAD THE GENE CHUNKS ---

	# set the gene chunk folder
	gene_chunk_folder_out = '%s/gene_chunks' % reg_head_folder

	# get the gene chunk paths
	genes_to_est_fileout_list = list(filter(lambda x: "genes_to_est" in x and ".txt" in x, os.listdir(gene_chunk_folder_out)))
	genes_to_est_fileout_list = sorted(genes_to_est_fileout_list, key=lambda x: int(x.replace(".txt","").split("_")[-1]))
	genes_to_est_fileout_list = list(map(lambda x: "%s/%s" % (gene_chunk_folder_out,x), genes_to_est_fileout_list))

	# get the chunk list
	chunk_gene_list = []
	for f in genes_to_est_fileout_list:
		with open(f) as file_obj:
			chunk = list(map(lambda x: x.replace("\n",""), file_obj.readlines()))
			chunk_gene_list.append(chunk_gene_list)





	# --- GET THE FOLDER OUTS ---


	# subfolders for each cluster / condition combo
	cluster_description_folder_out_dict = {}
	for cluster in clusters:
		cluster_reg_folder = '%s/cluster_%s' % (reg_head_folder,cluster)
		if not os.path.exists(cluster_reg_folder):
			os.makedirs(cluster_reg_folder)
		for description in descriptions:
			cluster_description_reg_folder = '%s/%s' % (cluster_reg_folder,description)
			if not os.path.exists(cluster_description_reg_folder):
				os.makedirs(cluster_description_reg_folder)
				
			# update dict
			if cluster not in cluster_description_folder_out_dict:
				cluster_description_folder_out_dict[cluster] = {}
			cluster_description_folder_out_dict[cluster][description] = cluster_description_reg_folder

		





	# --- CONCAT TOGETHER ---

	for cluster in clusters:
		for description in descriptions:

			de_novo_dfs, log_alpha_dfs, log_beta_dfs, min_max_dfs = [], [], [], []
			for chunk_index, gene_chunk in enumerate(chunk_gene_list):

				# get the path out
				path_out = cluster_description_folder_out_dict[cluster][description]
				path_out = "%s/chunk_%s" % (path_out,chunk_index)

				# load dfs
				de_novo_df = pd.read_table('%s/de_novo_metrics.tsv' % path_out,sep='\t',index_col='gene')
				log_alpha_df = pd.read_table('%s/gene_log_alpha.tsv' % path_out,sep='\t',index_col='gene')
				log_beta_df = pd.read_table('%s/gene_log_beta.tsv' % path_out,sep='\t',index_col='gene')
				min_max_df = pd.read_table('%s/log_min_max.tsv' % path_out,sep='\t',index_col='gene')

				# append to lists
				de_novo_dfs.append(de_novo_df)
				log_alpha_dfs.append(log_alpha_df)
				log_beta_dfs.append(log_beta_df)
				min_max_dfs.append(min_max_df)


			# ** concat **
			de_novo_df = pd.concat(de_novo_dfs)
			log_alpha_df = pd.concat(log_alpha_dfs)
			log_beta_df = pd.concat(log_beta_dfs)
			min_max_df = pd.concat(min_max_dfs)

			# ** write out **
			de_novo_df.to_csv('%s/de_novo_metrics.tsv' % cluster_description_folder_out_dict[cluster][description],sep='\t')
			log_alpha_df.to_csv('%s/gene_log_alpha.tsv' % cluster_description_folder_out_dict[cluster][description],sep='\t')
			log_beta_df.to_csv('%s/gene_log_beta.tsv' % cluster_description_folder_out_dict[cluster][description],sep='\t')
			min_max_df.to_csv('%s/log_min_max.tsv' % cluster_description_folder_out_dict[cluster][description],sep='\t')

			# ** cp the config from chunk 0 **
			command = 'cp "%s/chunk_0/config.txt" "%s/config.txt"' % (cluster_description_folder_out_dict[cluster][description],cluster_description_folder_out_dict[cluster][description])
			os.system(command)





if __name__ == "__main__":
	main(sys.argv)





