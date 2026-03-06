import sys
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import csv
import gzip
import scipy.io






# --- LOAD THE GENE SETS ---

def parse_gsea_set_file(gsea_set_path):
	
	# read lines
	with open(gsea_set_path) as file_obj:
		lines = file_obj.readlines()
		lines = list(map(lambda x: x.replace("\n",""), lines))

	# get gene set names and genes
	set_name_list = list(map(lambda x: x.split("\t")[0], lines))
	set_symbol_list = list(map(lambda x: x.split("\t")[2:], lines))

	# make it a dict
	gsea_dict = dict(zip(set_name_list,set_symbol_list))
	
	# lowercase and capitalize gene names
	for key, val in gsea_dict.items():
		gsea_dict[key] = list(map(lambda x: x.lower().capitalize(), gsea_dict[key]))

	return gsea_dict
	

def filter_gsea_dict(gsea_dict,gene_set):
	for key,val_list in gsea_dict.items():
		gsea_dict[key] = list(filter(lambda x: x in gene_set, val_list))
	keys_to_drop = []
	for key,val_list in gsea_dict.items():
		if len(val_list) == 0:
			keys_to_drop.append(key)
	print("Dropping %s keys" % len(keys_to_drop))
	for key in keys_to_drop:
		del gsea_dict[key]
		
	return gsea_dict


def load_gsea_dict(head_folder, gsea_set_name, genes_with_params):


	# set path
	if gsea_set_name not in ['gsea_location_gene_sets','gsea_kegg_pathways_gene_sets','gsea_reactome_pathways_gene_sets','go_biological_process']:
		raise Exception("Error: not a valid GSEA set name.")
	gsea_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/%s' % (head_folder,gsea_set_name)

	# load
	gsea_dict = parse_gsea_set_file(gsea_set_path)
	gsea_dict = filter_gsea_dict(gsea_dict,set(genes_with_params))

	return gsea_dict



def acrophase_enrichment(head_folder,fft_res,circadian_index,gene_set_dict,genes_with_params,num_grid_points):


	# --- LOAD GET SET DICT DICT ---

	# # set path
	# if gsea_set_name not in ['gsea_location_gene_sets','gsea_kegg_pathways_gene_sets','gsea_reactome_pathways_gene_sets','go_biological_process']:
	# 	raise Exception("Error: not a valid GSEA set name.")
	# gsea_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/%s' % (head_folder,gsea_set_name)

	# # load
	# gsea_dict = parse_gsea_set_file(gsea_set_path)
	# gsea_dict = filter_gsea_dict(gsea_dict,set(genes_with_params))


	# ** filter gene set dict **
	gene_set_dict = filter_gsea_dict(gene_set_dict,set(genes_with_params))

	# ** make index map **
	gene_name_index_map = dict(zip(genes_with_params,list(range(0,len(genes_with_params)))))




	# --- CONSTRUCT BIN LRT MAT ---


	gene_set_bin_lrt_mat = np.zeros((len(gene_set_dict),num_grid_points))
	for gene_set_index,gene_set in enumerate(list(gene_set_dict.keys())):
		
		# get the gene set indices
		gene_set_indices = list(map(lambda x: gene_name_index_map[x], gene_set_dict[gene_set]))
		
		# get acrophase samples
		acrophase_samples = np.arctan2(fft_res[:,circadian_index,gene_set_indices].flatten().imag, fft_res[:,circadian_index,gene_set_indices].flatten().real) % (2 * np.pi)

		# bin
		phases_discretized_indices = np.round((acrophase_samples / (2 * np.pi)) * num_grid_points) # convert [0,1] to [0,num_grid_points] (float), and then discretize
		phases_discretized_indices[np.where(phases_discretized_indices == num_grid_points)] = 0 # since bins actually go from [0,num_grid_points - 1], let's wrap back around
		phases_discretized_indices = phases_discretized_indices.astype(int)


		# get density
		bin_density = np.zeros((num_grid_points))
		for bin_index in np.arange(0,num_grid_points):
			bin_density[bin_index] = np.sum(phases_discretized_indices == bin_index)
		bin_density = bin_density / np.sum(bin_density)


		# get the density of uniform distribution
		uniform_val = 1.0 / num_grid_points

		# compute the LRT compared to the uniform
		gene_set_bin_lrt_mat[gene_set_index,:] = bin_density / uniform_val
		

	

	# --- MAKE THE BAYESIAN ENRICHMENT DF ---

	bayesian_gene_set_enrich_df = pd.DataFrame()
	bayesian_gene_set_enrich_df['gene_set'] = list(gene_set_dict.keys())
	bayesian_gene_set_enrich_df['num_genes'] = list(map(lambda x: len(x),list(gene_set_dict.values())))
	bayesian_gene_set_enrich_df['gene_set_bin_max_lrt'] = np.max(gene_set_bin_lrt_mat,axis=1)
	bayesian_gene_set_enrich_df = bayesian_gene_set_enrich_df.set_index("gene_set")
	bayesian_gene_set_enrich_df = bayesian_gene_set_enrich_df.sort_values('gene_set_bin_max_lrt',ascending=False)


	# --- MAKE THE LRT DF ----
	lrt_df = pd.DataFrame()
	lrt_df['gene_set'] = list(gene_set_dict.keys())
	for bin_index in range(0,num_grid_points):
		lrt_df['bin_%s' % bin_index] = gene_set_bin_lrt_mat[:,bin_index]
	lrt_df = lrt_df.set_index("gene_set")
	lrt_df = lrt_df.loc[bayesian_gene_set_enrich_df.index]

	
	return bayesian_gene_set_enrich_df,lrt_df, gene_set_dict








def gene_set_average_waveform(sampled_log_prop, gene_set_dict, gene_name_index_map):

	import torch


	# ** filter gene_set_dict **
	gene_set_dict = filter_gsea_dict(gene_set_dict,set(list(gene_name_index_map.keys())))


	# ** scale each sampled waveform from [0,1] **
	max_sampled_log_prop = torch.max(sampled_log_prop,dim=1).values.unsqueeze(1)
	min_sampled_log_prop = torch.min(sampled_log_prop,dim=1).values.unsqueeze(1)
	scaled_sampled_log_prop = (max_sampled_log_prop - sampled_log_prop) / (max_sampled_log_prop - min_sampled_log_prop)



	# ** get the gene set means waveforms **
	gene_set_sampled_waveforms = np.zeros((sampled_log_prop.shape[0], sampled_log_prop.shape[1], len(gene_set_dict)))
	for gene_set_index, gene_set in enumerate(list(gene_set_dict.keys())):

		# get the gene set indices
		gene_set_indices = list(map(lambda x: gene_name_index_map[x], gene_set_dict[gene_set]))

		# get gene_set_mean_waveform
		gene_set_sampled_waveforms[:,:,gene_set_index] = torch.mean(scaled_sampled_log_prop[:,:,gene_set_indices],dim=2).numpy()


	return gene_set_sampled_waveforms, gene_set_dict











