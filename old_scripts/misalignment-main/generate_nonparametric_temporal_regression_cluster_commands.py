import sys
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import anndata
import tqdm
import pandas as pd



# bsub -J job -e job.e -o job.o -M 64000 -R "rusage[mem=64000]" 'source /home/benauer/anaconda2/bin/activate tempo && python generate_nonparametric_temporal_regression_cluster_commands.py -f adata_qc_filtered.h5ad -cf clusters.tsv -o high_level_smc_chunk'

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




	# --- LOAD ADATA ---

	adata = anndata.read_h5ad(adata_path)


	# --- ADD EMBEDDINGS TO ADATA ---



	# ** load clusters **
	cluster_df = pd.read_table(cluster_path,sep='\t',index_col='index')

	# ** make sure everything in the same order
	adata = adata[list(cluster_df.index)]


	# ** add embeddings **
	print("printing the cluster df")
	print(cluster_df)

	try:
		adata.obs["cluster"] = np.array(cluster_df['leiden_scvi_cluster'])
	except:
		adata.obs["cluster"] = np.array(cluster_df['smc_subcluster'])





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





	# --- IDENTIFY GENES MEETING THE MINIMUM PROP IN EACH CLUSTER / CONDITION COMBO ---

	import warnings
	warnings.filterwarnings('ignore')

	genes_to_est = set()
	for i, description in enumerate(descriptions):
		for j, cluster in enumerate(clusters):
			print("Description: %s; Cluster: %s" % (i,j))
				
			cluster_condition_adata = adata[(adata.obs['cluster'] == cluster) & (adata.obs['description'] == description)]
			cluster_condition_adata.var['prop'] = np.array(np.sum(cluster_condition_adata.X,axis=0)).flatten()  / np.sum(cluster_condition_adata.obs['lib_size'])
			cluster_condition_adata = cluster_condition_adata[:,cluster_condition_adata.var['prop'] >= min_prop]
			genes_to_est.update(list(cluster_condition_adata.var_names))

	# make it a list
	genes_to_est = list(genes_to_est)


	# --- TEMP ---
	print("TEMPORARILY LIMITING TO CLUSTER 0")
	clusters = [0] 




	# --- MAKE THE GENE CHUNKS ---


	gene_chunk_indices = list(np.arange(0,len(genes_to_est),num_genes_per_chunk)) + [len(genes_to_est)]
	chunk_gene_list = []
	for gene_chunk_bin_index in range(0,len(gene_chunk_indices) - 1):
		chunk_start_index = gene_chunk_indices[gene_chunk_bin_index]
		chunk_end_index = gene_chunk_indices[gene_chunk_bin_index + 1]
		chunk_genes = genes_to_est[chunk_start_index:chunk_end_index]
		chunk_gene_list.append(chunk_genes)




	# --- MAKE THE FOLDER OUTS ---


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

		
		



	# --- WRITE OUT THE GENES TO ESTIMATE ---

	genes_to_est_fileout = '%s/genes_to_est.txt' % reg_head_folder
	with open(genes_to_est_fileout,"wb") as file_obj:
		file_obj.write("\n".join(genes_to_est).encode())





	# --- WRITE OUT THE GENE CHUNKS ---

	gene_chunk_folder_out = '%s/gene_chunks' % reg_head_folder
	if not os.path.exists(gene_chunk_folder_out):
		os.makedirs(gene_chunk_folder_out)
	for i, gene_chunk in enumerate(chunk_gene_list):
		genes_to_est_fileout = '%s/genes_to_est_%s.txt' % (gene_chunk_folder_out,i)
		with open(genes_to_est_fileout,"wb") as file_obj:
			file_obj.write("\n".join(gene_chunk).encode())




	# --- GET THE COMMANDS ---


	commands = []
	for cluster in clusters:
		for description in descriptions:
			for chunk_index, gene_chunk in enumerate(chunk_gene_list):
			
				# get the path out
				path_out = cluster_description_folder_out_dict[cluster][description]
				path_out = "%s/chunk_%s" % (path_out,chunk_index)

				# gene file
				gene_est_path = '%s/gene_chunks/genes_to_est_%s.txt' % (reg_head_folder, chunk_index)
				
				# get command
				command = "python run_non_parametric_reg.py -f %s -gf %s -cf %s -c %s -d '%s' -o '%s'" % (adata_path, gene_est_path, cluster_path, cluster, description, path_out)

				# add conda activate to command
				command = "source /home/benauer/anaconda2/bin/activate tempo && %s" % command

				# make the job
				job_string = '%s_%s_chunk_%s' % (cluster,description,chunk_index)
				job_string = job_string.replace(" ", "_")
				command = 'bsub -J %s -M 64000 -e error_files/%s.e -o out_files/%s.o -R "rusage[mem=64000]" "%s"' % (job_string,job_string,job_string, command)

				# add
				commands.append(command)
			
			




	# --- WRITE THE COMMANDS ---

	with open('reg_commands.sh',"wb") as file_obj:
		file_obj.write("\n".join(commands).encode())










if __name__ == "__main__":
	main(sys.argv)






