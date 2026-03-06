

import sys
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import csv
import gzip
import scipy.io
import numpy as np 
import pandas as pd
import anndata
import torch



# PLOTTING STUFF
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import seaborn as sns
from matplotlib import rc
rc_params = {
	"font.family": "sans-serif", # "text.usetex": True,
	"font.serif" : ['Arial'], # "font.size" : 6,
	"font.weight" : "normal",
	'xtick.labelsize' : 5,
	'ytick.labelsize' : 5,
	'legend.fontsize' : 5,
	'axes.labelsize' : 6,
	'axes.linewidth': 0.8,
	'lines.linewidth' : 1,
	'axes.titlesize' : 7 # figure.titlesize
}
matplotlib.rcParams.update(rc_params)


# --- DEFINE FUNCTION TO ADD TRANSFOREMD X TO ADATA ---

def add_transformed_X_to_adata(adata,pseudocount=1e-7):
	X_transformed = np.array(adata.X.todense()) / np.array(adata.obs['lib_size']).reshape(-1,1)
	X_transformed[X_transformed <= pseudocount] = pseudocount
	X_transformed = np.log10(X_transformed)
	adata.layers['X_transformed'] = X_transformed
	return adata




# --- FN TO MAKE LOW-D CELL DENSITY PLOT ----
def viz_cell_density(x_vals,y_vals,x_col,y_col,cmap,title=None,xlim_tuple=None,ylim_tuple=None,dpi=200,ax=None,show=True):
	

	# ** get DF **
	density_df = pd.DataFrame()
	density_df[x_col] = x_vals
	density_df[y_col] = y_vals
	
	# ** plot **
	
	# init
	if ax is None:
		fig,ax = plt.subplots(dpi = dpi)
	
	# set figure limits if supplied
	if xlim_tuple:
		plt.xlim(xlim_tuple)
	if ylim_tuple:
		plt.ylim(ylim_tuple)

		
	# set xlim and ylim box to the minimum cmap value
	if xlim_tuple is not None and ylim_tuple is not None:
		
		# get minimum CMAP value
		min_cmap_val = matplotlib.cm.get_cmap(cmap)(0.0) # cmap = matplotlib.cm.get_cmap(cmap)
		# min_cmap_val = cmap(0.0)
		box_points_x = [x_min, x_min, x_max, x_max]
		box_points_y = [y_min, y_max,y_max,y_min]
		plt.fill(box_points_x, box_points_y, c = min_cmap_val, alpha = 1.0)
	
	# plot
	ax = sns.kdeplot(data=density_df, x=x_col, y=y_col,ax=ax,
					fill=True, thresh=0, levels=100, cmap=cmap)
	if title is not None:
		plt.title(title)
	if show:
		plt.show()
	
	return ax






def get_log_prop_samples(cluster,description,num_samples,cluster_description_folder_out_dict,gene_list=None):
	
	# ** load metric df **
	metric_path = '%s/de_novo_metrics.tsv' % cluster_description_folder_out_dict[cluster][description]
	metric_df = pd.read_table(metric_path,index_col='gene',sep='\t')

	# ** load gene parameters **
	gene_log_alpha_path = '%s/gene_log_alpha.tsv' % cluster_description_folder_out_dict[cluster][description]
	gene_log_beta_path = '%s/gene_log_beta.tsv' % cluster_description_folder_out_dict[cluster][description]
	gene_log_min_max_path = '%s/log_min_max.tsv' % cluster_description_folder_out_dict[cluster][description]
	gene_log_alpha_df = pd.read_table(gene_log_alpha_path,sep='\t',index_col='gene')
	gene_log_beta_df = pd.read_table(gene_log_beta_path,sep='\t',index_col='gene')
	gene_log_min_max_df = pd.read_table(gene_log_min_max_path,sep='\t',index_col='gene')
	gene_list = list(filter(lambda x: x in gene_log_alpha_df.index,gene_list))
	if gene_list is not None:
		gene_log_alpha_df = gene_log_alpha_df.loc[gene_list]
		gene_log_beta_df = gene_log_beta_df.loc[gene_list]
		gene_log_min_max_df = gene_log_min_max_df.loc[gene_list]
	gene_log_min = torch.Tensor(np.array(gene_log_min_max_df['log_min'])) 
	gene_log_max = torch.Tensor(np.array(gene_log_min_max_df['log_max'])) 
	
	
	# ** get the log_prop_dist obj **
	log_prop_dist = torch.distributions.beta.Beta(torch.Tensor(np.exp(np.array(gene_log_alpha_df)).T), torch.Tensor(np.exp(np.array(gene_log_beta_df))).T)

	# ** sample the waveforms **
	sampled_log_prop = log_prop_dist.sample((num_samples,)) # [num_samples x num_time_points x num_genes]
	sampled_log_prop = (sampled_log_prop * (gene_log_max.unsqueeze(0).unsqueeze(0) - gene_log_min.unsqueeze(0).unsqueeze(0))) + gene_log_min.unsqueeze(0).unsqueeze(0)
	
	
	return sampled_log_prop,gene_list





# --- DEFINE FUNCTION TO GET METRICS RELEVANT FOR THE DIFFERENCE BETWEEN WAVEFORMS ---



# samples are of shape: [num_samples x num_genes]
def compute_fraction_samples_crossing_zero_mean(samples):

	# get expected values for each gene
	expected_vals = np.mean(samples,axis=0)

	# get the indices of the genes with negative expected values
	neg_ev_gene_indices = np.where(expected_vals < 0)[0]

	# compute the fraction of samples crossing zero for genes with positive EV
	frac_samples_crossing_zero = np.mean(samples < 0,axis=0)

	# compute the fraction of samples crossing zero for genes with negative EV
	frac_samples_crossing_zero[neg_ev_gene_indices] = np.mean(samples[:,neg_ev_gene_indices] > 0, axis=0)

	return frac_samples_crossing_zero



# samples are of shape: [num_samples x num_genes]
def compute_fraction_of_amp_samples_crossing_zero(real_samples, imag_samples):

	# compute the fraction of real valued circadian samples above zero
	frac_real_circadian_samples_crossing_zero = compute_fraction_samples_crossing_zero_mean(real_samples)

	# compute the fraction of imag valued circadian samples above zero
	frac_imag_circadian_samples_crossing_zero = compute_fraction_samples_crossing_zero_mean(imag_samples)

	# compute fraction of circadian samples crossing zero
	return np.min(np.concatenate((frac_real_circadian_samples_crossing_zero.reshape(-1,1),frac_imag_circadian_samples_crossing_zero.reshape(-1,1)),axis=1),axis=1)



def mesor_dif_metrics(waveform_sampled_1, waveform_sampled_2):
	
	   
	# get mesor samples
	mesor_samples_1 = torch.mean(waveform_sampled_1,dim=1).numpy()
	mesor_samples_2 = torch.mean(waveform_sampled_2,dim=1).numpy()

	# compute the fraction greater / smaller in the 2nd samples
	frac_mesor_samples_larger = np.mean(mesor_samples_2 > mesor_samples_1,axis=0)

	# compute the effect size
	mesor_effect = np.mean(mesor_samples_2,axis=0) - np.mean(mesor_samples_1,axis=0)

	return frac_mesor_samples_larger, mesor_effect


def amp_dif_metrics(fft_sampled_1, fft_sampled_2, circadian_index):
	
	# get amp samples
	amp_samples_1 = abs(fft_sampled_1)[:,circadian_index.item(),:]
	amp_samples_2 = abs(fft_sampled_2)[:,circadian_index.item(),:]

	# compute the fraction greater / smaller in the 2nd samples
	frac_amp_samples_larger = np.mean(amp_samples_2 > amp_samples_1,axis=0)

	# compute the effect size
	amp_effect = np.mean((amp_samples_2 - amp_samples_1),axis=0)

	return frac_amp_samples_larger, amp_effect


def acrophase_dif_metrics(fft_sampled_1, fft_sampled_2, circadian_index):
	

	# ** get euclid acrophase samples 1 **
	acrophase_samples_1 = (np.arctan2(fft_sampled_1[:,circadian_index.item(),:].imag,fft_sampled_1[:,circadian_index.item(),:].real) % (2 * np.pi))
	acrophase_euclid_samples_1 = np.zeros((acrophase_samples_1.shape[0],acrophase_samples_1.shape[1],2))
	acrophase_euclid_samples_1[:,:,0] = np.cos(acrophase_samples_1)
	acrophase_euclid_samples_1[:,:,1] = np.sin(acrophase_samples_1)

	# ** get euclid acrophase samples 2 **
	acrophase_samples_2 = (np.arctan2(fft_sampled_2[:,circadian_index.item(),:].imag,fft_sampled_2[:,circadian_index.item(),:].real) % (2 * np.pi))
	acrophase_euclid_samples_2 = np.zeros((acrophase_samples_2.shape[0],acrophase_samples_2.shape[1],2))
	acrophase_euclid_samples_2[:,:,0] = np.cos(acrophase_samples_2)
	acrophase_euclid_samples_2[:,:,1] = np.sin(acrophase_samples_2)

	
	# ** compute the dif acrophase samples **
	dif_acrophase_euclid_samples = acrophase_euclid_samples_2 - acrophase_euclid_samples_1

	# ** compute the fraction of non-zero acrophase difference samples **
	frac_zero_acrophase_dif_samples = compute_fraction_of_amp_samples_crossing_zero(dif_acrophase_euclid_samples[:,:,0], dif_acrophase_euclid_samples[:,:,1])

	# ** get the mean acrophase effect **
	expected_acrophase_1 = (2 * np.pi - scipy.stats.circmean(acrophase_samples_1,high=2*np.pi,low=0,axis=0)) % (2 * np.pi)
	expected_acrophase_2 = (2 * np.pi - scipy.stats.circmean(acrophase_samples_2,high=2*np.pi,low=0,axis=0)) % (2 * np.pi)
	acrophase_effect = expected_acrophase_2 - expected_acrophase_1
	acrophase_effect = acrophase_effect % (2 * np.pi)
	acrophase_effect[acrophase_effect >= np.pi] = -1 * ((2 * np.pi) - acrophase_effect[acrophase_effect >= np.pi])

	
	return frac_zero_acrophase_dif_samples, acrophase_effect 



def acrophase_dif_metrics_2(fft_sampled_1, fft_sampled_2, circadian_index,hour_thresh=3):
	# ** get euclid acrophase samples 1 **
	acrophase_samples_1 = (np.arctan2(fft_sampled_1[:,circadian_index.item(),:].imag,fft_sampled_1[:,circadian_index.item(),:].real) % (2 * np.pi))
	acrophase_samples_1 = ((2 * np.pi) - acrophase_samples_1) % (2 * np.pi)

	# ** get euclid acrophase samples 2 **
	acrophase_samples_2 = (np.arctan2(fft_sampled_2[:,circadian_index.item(),:].imag,fft_sampled_2[:,circadian_index.item(),:].real) % (2 * np.pi))
	acrophase_samples_2 = ((2 * np.pi) - acrophase_samples_2) % (2 * np.pi)

	# ** compute the dif acrophase samples **
	dif_acrophase_samples = (acrophase_samples_2 - acrophase_samples_1) % (2 * np.pi)

	# ** compute the fraction of acrophase dif samples less than the threshold **
	fraction_acrophase_dif_samples_mag_less_than_thresh = np.mean((dif_acrophase_samples >= ((24 - hour_thresh) / 24) * 2 * np.pi) | (dif_acrophase_samples <= (hour_thresh / 24) * 2 * np.pi),axis=0)
	

	# ** get the expected acrophase effect **
	expected_acrophase_effect = scipy.stats.circmean(dif_acrophase_samples,low=0,high=2*np.pi,axis=0)


	# vals = np.mean((dif_acrophase_samples >= (4 / 24) * 2 * np.pi) & (dif_acrophase_samples <= (8 / 24) * 2 * np.pi),axis=0)
	# waveform_dif_df['acrophase_effect'] = np.array(waveform_dif_df['acrophase_effect']) % (2 * np.pi)
	# is_shifted = np.array(gene_stat_df_1['frac_circadian_samples_crossing_zero'] <= frac_circadian_samples_crossing_zero_thresh) & np.array(gene_stat_df_2['frac_circadian_samples_crossing_zero'] <= frac_circadian_samples_crossing_zero_thresh) & np.array(waveform_dif_df['frac_zero_acrophase_dif_samples'] <= acrophase_frac_thresh)
	# is_shifted_4_to_8 = is_shifted & np.array((waveform_dif_df['acrophase_effect'] >= (4 / 24) * 2 * np.pi) & (waveform_dif_df['acrophase_effect'] <= (8 / 24) * 2 * np.pi))
	# is_shifted_8_to_12 = is_shifted & np.array((waveform_dif_df['acrophase_effect'] >= (8 / 24) * 2 * np.pi) & (waveform_dif_df['acrophase_effect'] <= (12 / 24) * 2 * np.pi))
	# is_shifted_12_to_16 = is_shifted & np.array((waveform_dif_df['acrophase_effect'] >= (12 / 24) * 2 * np.pi) & (waveform_dif_df['acrophase_effect'] <= (16 / 24) * 2 * np.pi))
	# is_shifted_16_to_20 = is_shifted & np.array((waveform_dif_df['acrophase_effect'] >= (16 / 24) * 2 * np.pi) & (waveform_dif_df['acrophase_effect'] <= (20 / 24) * 2 * np.pi))



	return fraction_acrophase_dif_samples_mag_less_than_thresh,expected_acrophase_effect, dif_acrophase_samples






def cosinor_dif_metrics(fft_sampled_1, fft_sampled_2, circadian_index):
	
	# ** get cosinor coefficient samples 1 **
	cosinor_euclid_samples_1 = np.zeros((fft_sampled_1.shape[0],fft_sampled_1.shape[2],2))
	cosinor_euclid_samples_1[:,:,0] = fft_sampled_1[:,circadian_index.item(),:].real
	cosinor_euclid_samples_1[:,:,1] = fft_sampled_1[:,circadian_index.item(),:].imag
	cosinor_acrophase_samples_1 = (np.arctan2(fft_sampled_1[:,circadian_index.item(),:].imag,fft_sampled_1[:,circadian_index.item(),:].real) % (2 * np.pi))
	
	
	# ** get cosinor coefficient samples 2 **
	cosinor_euclid_samples_2 = np.zeros((fft_sampled_1.shape[0],fft_sampled_1.shape[2],2))
	cosinor_euclid_samples_2[:,:,0] = fft_sampled_2[:,circadian_index.item(),:].real
	cosinor_euclid_samples_2[:,:,1] = fft_sampled_2[:,circadian_index.item(),:].imag
	cosinor_acrophase_samples_2 = (np.arctan2(fft_sampled_2[:,circadian_index.item(),:].imag,fft_sampled_2[:,circadian_index.item(),:].real) % (2 * np.pi))
	
	
	# ** compute the dif acrophase samples **
	dif_cosinor_euclid_samples = cosinor_euclid_samples_2 - cosinor_euclid_samples_1

	# ** compute the fraction of non-zero cosinor difference samples **
	frac_zero_cosinor_dif_samples = compute_fraction_of_amp_samples_crossing_zero(dif_cosinor_euclid_samples[:,:,0], dif_cosinor_euclid_samples[:,:,1])

	# ** get the mean cosinor euclid effect **
	cosinor_euclid_effect = np.mean(cosinor_euclid_samples_2,axis=0) - np.mean(cosinor_euclid_samples_1,axis=0)
	
	return frac_zero_cosinor_dif_samples, cosinor_euclid_effect 





def load_param_dfs_from_path(res_folder):
	gene_log_alpha_path = '%s/gene_log_alpha.tsv' % res_folder
	gene_log_beta_path = '%s/gene_log_beta.tsv' % res_folder
	gene_log_min_max_path = '%s/log_min_max.tsv' % res_folder
	gene_stat_path = '%s/de_novo_metrics.tsv' % res_folder
	gene_log_alpha_df = pd.read_table(gene_log_alpha_path,sep='\t',index_col='gene')
	gene_log_beta_df = pd.read_table(gene_log_beta_path,sep='\t',index_col='gene')
	gene_log_min_max_df = pd.read_table(gene_log_min_max_path,sep='\t',index_col='gene')
	gene_stat_df = pd.read_table(gene_stat_path,sep='\t',index_col='gene')
	return gene_log_alpha_df, gene_log_beta_df, gene_log_min_max_df,gene_stat_df






# --- MOTIF ENRICHMENT ---
import statsmodels
from statsmodels import stats
from statsmodels.stats import multitest
def compute_motif_enrichment(motif_lrt_df, cluster_cyclers, background_genes, num_shuffles = 1000):
	
	# get cluster cyclers in the motif df
	cluster_cyclers = list(filter(lambda x: x in motif_lrt_df.index,cluster_cyclers))

	# get the background gene set in the motif df
	background_genes = list(filter(lambda x: x in motif_lrt_df.index,background_genes))

	# get the cycler stat: [num_tf_motifs]
	cycler_stat = np.array(np.sum(motif_lrt_df.loc[cluster_cyclers],axis=0))

	# num shuffles
	null_dist_mat = np.zeros((len(cycler_stat),num_shuffles))
	for shuffle in range(0,num_shuffles):
		random_genes = np.random.choice(background_genes,size=len(cluster_cyclers),replace=True)
		null_stat = np.array(np.sum(motif_lrt_df.loc[random_genes],axis=0))
		null_dist_mat[:,shuffle] = null_stat


	# compute the p vals
	motif_p_vals = []
	for motif_index in range(0,len(cycler_stat)):
		p_val = (100 - scipy.stats.percentileofscore(null_dist_mat[motif_index,:],cycler_stat[motif_index])) / 100.0
		motif_p_vals.append(p_val)

	# compute p val df
	motif_p_val_df = pd.DataFrame()
	motif_p_val_df['motif'] = list(motif_lrt_df.columns)
	motif_p_val_df['p_val'] = motif_p_vals
	motif_p_val_df['cycler_stat'] = cycler_stat
	motif_p_val_df['null_1st_percentile'] = np.percentile(null_dist_mat,1,axis=1)
	motif_p_val_df['null_5th_percentile'] = np.percentile(null_dist_mat,5,axis=1)
	motif_p_val_df['null_50th_percentile'] = np.percentile(null_dist_mat,50,axis=1)
	motif_p_val_df['null_95th_percentile'] = np.percentile(null_dist_mat,95,axis=1)
	motif_p_val_df['null_99th_percentile'] = np.percentile(null_dist_mat,99,axis=1)
	motif_p_val_df = motif_p_val_df.set_index("motif")



	# ** get BH corrected p-values **
	motif_p_val_df['bh_q'] = statsmodels.stats.multitest.fdrcorrection(np.array(motif_p_val_df['p_val']), alpha=0.05, method='indep', is_sorted=False)[1]
	
	return motif_p_val_df




# --- GENE SET ENRICHMENT OVER TIME ---
def get_enrichment_over_time(genes_over_time, background_genes, enrichment_gene_set = 'KEGG_2016', signif_thresh = 0.1):
	import gseapy
	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)
	
	
	# ** get the number of time points **
	num_time_points = len(genes_over_time)


	# ** init the signif genes dict **
	signif_term_genes_dict = {}
	

	# ** get enrichment results at each time point **
	enrichment_res_dfs_per_time = []
	for time_index in range(0,num_time_points):

		print(time_index)

		# run
		res = gseapy.enrichr(gene_list=genes_over_time[time_index],
							 gene_sets = enrichment_gene_set,
							organism = 'mouse',
							background=background_genes,
							cutoff=1.0)


		# get the DF
		res_df = res.results
		res_df = res_df.set_index("Term")

		# convert genes to mouse format (from human)
		res_df['Genes'] = list(map(lambda y: ";".join(list(map(lambda x: x.lower().capitalize(), y.split(";")))), list(res_df['Genes'])))

		# add
		enrichment_res_dfs_per_time.append(res_df)

		# update signif_term_genes_dict
		signif_res_df = res_df[res_df['Adjusted P-value'] <= signif_thresh]
		for term in signif_res_df.index:
			if term not in signif_term_genes_dict:
				signif_term_genes_dict[term] = set()
			signif_term_genes_dict[term].update(set(signif_res_df.loc[term]['Genes'].split(";")))





	# ** get the unique terms **
	unique_terms = set()
	for enrichment_res_df_per_time in enrichment_res_dfs_per_time:
		unique_terms.update(set(enrichment_res_df_per_time.index))
	unique_terms = sorted(list(unique_terms))




	# ** fill missing terms in each **
	for i, enrichment_res_df_per_time in enumerate(enrichment_res_dfs_per_time):

		# get the missing terms 
		missing_terms = set(unique_terms).difference(set(enrichment_res_dfs_per_time[i].index))

		# init new DF
		new_df = pd.DataFrame()

		# fill relevant columns
		new_df['Term'] = list(missing_terms)
		new_df['Adjusted P-value'] = 1.0
		new_df['Genes'] = ''
		new_df = new_df.set_index("Term")
		
		# fill columns w/ empty vals
		for col in enrichment_res_df_per_time.columns:
			if col not in ['Adjusted P-value','Term','Genes']:
				new_df[col] = np.nan



		# update the enrichment df
		enrichment_res_dfs_per_time[i] = pd.concat((enrichment_res_df_per_time,new_df),axis=0)

		# make sure all of them are in the same order
		enrichment_res_dfs_per_time[i] = enrichment_res_dfs_per_time[i].loc[unique_terms]





	# ** make the enrichment_time_mat **
	enrichment_time_mat = np.ones((len(unique_terms),num_time_points))
	for time_index in range(0,num_time_points):

		# add
		enrichment_time_mat[:,time_index] = np.array(enrichment_res_dfs_per_time[time_index]['Adjusted P-value'])


	# ** make the enrichment_time_df **
	enrichment_time_df = pd.DataFrame()
	enrichment_time_df['ZT0'] = enrichment_time_mat[:,0]
	enrichment_time_df['ZT6'] = enrichment_time_mat[:,1]
	enrichment_time_df['ZT12'] = enrichment_time_mat[:,2]
	enrichment_time_df['ZT18'] = enrichment_time_mat[:,3]
	enrichment_time_df['Term'] = unique_terms
	enrichment_time_df = enrichment_time_df.set_index("Term")


	# ** get the signif df **
	signif_enrichment_time_df = enrichment_time_df.loc[np.min(enrichment_time_df,axis=1) <= signif_thresh]




	# return enrichment_time_df, signif_enrichment_time_df, enrichment_res_dfs_per_time
	return enrichment_res_dfs_per_time, enrichment_time_df, signif_enrichment_time_df, signif_term_genes_dict
	




def run_enrichment_over_time(up_genes_over_time,down_genes_over_time,enrichment_gene_set,genes_with_params,folder_out,cluster_cell_type_dict,cluster,sampled_log_prop,signif_thresh=0.1):

	# ** get up enrichment df **
	up_enrichment_res_dfs_per_time, up_enrichment_time_df, up_signif_enrichment_time_df, up_signif_term_genes_dict = get_enrichment_over_time(genes_over_time = up_genes_over_time, 
							 background_genes = list(genes_with_params), 
							 enrichment_gene_set = enrichment_gene_set, signif_thresh = signif_thresh)



	# ** get down enrichment df **
	down_enrichment_res_dfs_per_time, down_enrichment_time_df, down_signif_enrichment_time_df, down_signif_term_genes_dict = get_enrichment_over_time(genes_over_time = down_genes_over_time, 
							 background_genes = list(genes_with_params), 
							 enrichment_gene_set = enrichment_gene_set, signif_thresh = signif_thresh)


	# ** merge the signif terms dict **
	import copy
	signif_terms_gene_dict = copy.deepcopy(up_signif_term_genes_dict)
	for term in down_signif_term_genes_dict:
		if term not in signif_terms_gene_dict:
			signif_terms_gene_dict[term] = down_signif_term_genes_dict[term]
		else:
			signif_terms_gene_dict[term].update(down_signif_term_genes_dict[term])

			
	# ** get the missing terms in the up and down **
	terms_not_in_down = set(up_enrichment_time_df.index).difference(set(down_enrichment_time_df.index))
	terms_not_in_up = set(down_enrichment_time_df.index).difference(set(up_enrichment_time_df.index))
	union_terms = set(up_enrichment_time_df.index).union(set(down_enrichment_time_df.index))


	# ** fill down DF w/ the terms not in down **
	new_df = pd.DataFrame(data = np.ones((len(terms_not_in_down),4)))
	new_df['Term'] = list(terms_not_in_down)
	new_df = new_df.set_index("Term")
	new_df.columns = ['ZT0','ZT6','ZT12','ZT18']
	down_enrichment_time_df = pd.concat((down_enrichment_time_df,new_df),axis=0)
	down_enrichment_time_df = down_enrichment_time_df.loc[list(union_terms)] # put in same order


	# ** fill up DF w/ the terms not in up **
	new_df = pd.DataFrame(data = np.ones((len(terms_not_in_up),4)))
	new_df['Term'] = list(terms_not_in_up)
	new_df = new_df.set_index("Term")
	new_df.columns = ['ZT0','ZT6','ZT12','ZT18']
	up_enrichment_time_df = pd.concat((up_enrichment_time_df,new_df),axis=0)
	up_enrichment_time_df = up_enrichment_time_df.loc[list(union_terms)] # put in same order


	# ** find the pathways that show circadian rhythms in enrichment **

	# get the candidate signif terms
	candidate_signif_terms = list(set(up_signif_enrichment_time_df.index).union(set(down_signif_enrichment_time_df.index)))

	# code down
	down_enrichment_time_coded = np.array(down_enrichment_time_df.loc[candidate_signif_terms])
	down_enrichment_time_coded[np.array(down_enrichment_time_df.loc[candidate_signif_terms]) <= 0.2] = -1
	down_enrichment_time_coded[(np.array(down_enrichment_time_df.loc[candidate_signif_terms]) >= 0.2) & (np.array(down_enrichment_time_df.loc[candidate_signif_terms]) <= 0.8)] = 0
	down_enrichment_time_coded[np.array(down_enrichment_time_df.loc[candidate_signif_terms]) >= 0.8] = 0

	# code up
	up_enrichment_time_coded = np.array(up_enrichment_time_df.loc[candidate_signif_terms])
	up_enrichment_time_coded[up_enrichment_time_df.loc[candidate_signif_terms] <= 0.2] = 1
	up_enrichment_time_coded[(up_enrichment_time_df.loc[candidate_signif_terms] >= 0.2) & (up_enrichment_time_df.loc[candidate_signif_terms] <= 0.8)] = 0
	up_enrichment_time_coded[up_enrichment_time_df.loc[candidate_signif_terms] >= 0.8] = 0


	# ** get the consistent df **
	consistent_df = pd.DataFrame()
	consistent_df['term'] = candidate_signif_terms
	consistent_df['consistent_direction'] = np.min(up_enrichment_time_coded * down_enrichment_time_coded,axis=1) != -1
	consistent_df = consistent_df.set_index("term")


	# ** get consistent terms **
	consistent_terms = list(consistent_df[consistent_df['consistent_direction']].index)


	# ** write df results out **

	# make the subfolder
	subfolder_out = '%s/cycler_enrichment_over_time/%s/%s' % (folder_out,enrichment_gene_set,cluster_cell_type_dict[cluster])
	if not os.path.exists(subfolder_out):
		os.makedirs(subfolder_out)
		
	# write out up enrichment
	fileout = '%s/up_enrichment_time.tsv' % subfolder_out
	up_enrichment_time_df.to_csv(fileout,sep='\t')
		
	# write out signif up enrichment
	fileout = '%s/up_signif_enrichment_time.tsv' % subfolder_out
	up_signif_enrichment_time_df.to_csv(fileout,sep='\t')


	# write out down enrichment
	fileout = '%s/down_enrichment_time.tsv' % subfolder_out
	down_enrichment_time_df.to_csv(fileout,sep='\t')
		
	# write out signif down enrichment
	fileout = '%s/down_signif_enrichment_time.tsv' % subfolder_out
	down_signif_enrichment_time_df.to_csv(fileout,sep='\t')
		
		
	# write out consistent df
	fileout = '%s/consistent_signif_enrichment_time.tsv' % subfolder_out
	consistent_df.to_csv(fileout,sep='\t')
		

	# write out down consistent df
	fileout = '%s/down_consistent_signif_enrichment_time.tsv' % subfolder_out
	down_enrichment_time_df.loc[consistent_terms].to_csv(fileout,sep='\t')


	# write out up consistent df
	fileout = '%s/up_consistent_signif_enrichment_time.tsv' % subfolder_out
	up_enrichment_time_df.loc[consistent_terms].to_csv(fileout,sep='\t')


	# make visualizations of up enrichments over time
	for term in up_enrichment_time_df.index:
	    fig,ax=plt.subplots(figsize=(2,1.6),dpi=300)
	    sns.despine()
	    plt.scatter(['ZT0', 'ZT6', 'ZT12', 'ZT18'], np.array(up_enrichment_time_df.loc[term]),s=5)
	    plt.ylabel("Overrepresentation p-value")
	    viz_subfolder_out = '%s/up_enrichment_plots' % (subfolder_out)
	    if not os.path.exists(viz_subfolder_out):
	        os.makedirs(viz_subfolder_out)
	    fileout = '%s/%s.pdf' % (viz_subfolder_out,term.replace("/", "-"))
	    plt.title(term)
	    plt.ylim((-0.1,1.1))
	    plt.savefig(fileout,bbox_inches='tight')
		


	# make visualizations of down enrichments over time
	for term in down_enrichment_time_df.index:
	    fig,ax=plt.subplots(figsize=(2,1.6),dpi=300)
	    sns.despine()
	    plt.scatter(['ZT0', 'ZT6', 'ZT12', 'ZT18'], np.array(down_enrichment_time_df.loc[term]),s=5)
	    plt.ylabel("Overrepresentation p-value")
	    viz_subfolder_out = '%s/down_enrichment_plots' % (subfolder_out)
	    if not os.path.exists(viz_subfolder_out):
	        os.makedirs(viz_subfolder_out)
	    fileout = '%s/%s.pdf' % (viz_subfolder_out,term.replace("/", "-"))
	    plt.title(term)
	    plt.ylim((-0.1,1.1))
	    plt.savefig(fileout,bbox_inches='tight')



	# # ** viz genes in the consistent set and write out examples **
	# for term in up_enrichment_time_df.index:
	# 	for gene in up_signif_term_genes_dict[term]:


	# 		gene_index = np.where(np.array(genes_with_params) == gene)[0].item()


	# 		# get min and max vals for the gene
	# 		min_val = torch.min(sampled_log_prop[:,:,gene_index].flatten()).item() / np.log(10)
	# 		max_val = torch.max(sampled_log_prop[:,:,gene_index].flatten()).item() / np.log(10)


	# 		# make gene df
	# 		val_mat = sampled_log_prop[:,:,gene_index]
	# 		time_mat = np.zeros((sampled_log_prop.shape[0], 4))
	# 		time_mat[:,0] = 0
	# 		time_mat[:,1] = 6
	# 		time_mat[:,2] = 12
	# 		time_mat[:,3] = 18
	# 		gene_df = pd.DataFrame()
	# 		gene_df['ZT'] = time_mat.flatten().astype(int)
	# 		gene_df['Log10 prop'] = val_mat.flatten().numpy() / np.log(10)


	# 		# viz
	# 		fig,ax=plt.subplots(figsize=(2,1.6),dpi=300)
	# 		sns.despine()
	# 		sns.violinplot(x="ZT", y="Log10 prop", data=gene_df,ax=ax,color='0.5')
	# 		viz_subfolder_out = '%s/up_enriched_term_example_genes/%s' % (subfolder_out,term)
	# 		if not os.path.exists(viz_subfolder_out):
	# 			os.makedirs(viz_subfolder_out)
	# 		fileout = '%s/%s.pdf' % (viz_subfolder_out,gene)
	# 		plt.savefig(fileout,bbox_inches='tight')
	# 		plt.title(gene)
	# 		# plt.show()



	# # ** viz genes in the consistent set and write out examples **
	# for term in down_enrichment_time_df.index:
	# 	for gene in down_signif_term_genes_dict[term]:


	# 		gene_index = np.where(np.array(genes_with_params) == gene)[0].item()


	# 		# get min and max vals for the gene
	# 		min_val = torch.min(sampled_log_prop[:,:,gene_index].flatten()).item() / np.log(10)
	# 		max_val = torch.max(sampled_log_prop[:,:,gene_index].flatten()).item() / np.log(10)


	# 		# make gene df
	# 		val_mat = sampled_log_prop[:,:,gene_index]
	# 		time_mat = np.zeros((sampled_log_prop.shape[0], 4))
	# 		time_mat[:,0] = 0
	# 		time_mat[:,1] = 6
	# 		time_mat[:,2] = 12
	# 		time_mat[:,3] = 18
	# 		gene_df = pd.DataFrame()
	# 		gene_df['ZT'] = time_mat.flatten().astype(int)
	# 		gene_df['Log10 prop'] = val_mat.flatten().numpy() / np.log(10)


	# 		# viz
	# 		fig,ax=plt.subplots(figsize=(2,1.6),dpi=300)
	# 		sns.despine()
	# 		sns.violinplot(x="ZT", y="Log10 prop", data=gene_df,ax=ax,color='0.5')
	# 		viz_subfolder_out = '%s/down_enriched_term_example_genes/%s' % (subfolder_out,term)
	# 		if not os.path.exists(viz_subfolder_out):
	# 			os.makedirs(viz_subfolder_out)
	# 		fileout = '%s/%s.pdf' % (viz_subfolder_out,gene)
	# 		plt.savefig(fileout,bbox_inches='tight')
	# 		plt.title(gene)
	# 		# plt.show()



	# ** viz genes in the consistent set and write out examples **
	for term in consistent_terms:
		for gene in signif_terms_gene_dict[term]:


			gene_index = np.where(np.array(genes_with_params) == gene)[0].item()


			# get min and max vals for the gene
			min_val = torch.min(sampled_log_prop[:,:,gene_index].flatten()).item() / np.log(10)
			max_val = torch.max(sampled_log_prop[:,:,gene_index].flatten()).item() / np.log(10)


			# make gene df
			val_mat = sampled_log_prop[:,:,gene_index]
			time_mat = np.zeros((sampled_log_prop.shape[0], 4))
			time_mat[:,0] = 0
			time_mat[:,1] = 6
			time_mat[:,2] = 12
			time_mat[:,3] = 18
			gene_df = pd.DataFrame()
			gene_df['ZT'] = time_mat.flatten().astype(int)
			gene_df['Log10 prop'] = val_mat.flatten().numpy() / np.log(10)


			# viz
			fig,ax=plt.subplots(figsize=(2,1.6),dpi=300)
			sns.despine()
			sns.violinplot(x="ZT", y="Log10 prop", data=gene_df,ax=ax,color='0.5')
			viz_subfolder_out = '%s/enriched_term_example_genes/%s' % (subfolder_out,term)
			if not os.path.exists(viz_subfolder_out):
				os.makedirs(viz_subfolder_out)
			fileout = '%s/%s.pdf' % (viz_subfolder_out,gene)
			plt.savefig(fileout,bbox_inches='tight')
			plt.title(gene)
			# plt.show()




def get_gene_set_dicts(head_folder,reg_genes_to_est=None):

	# --- GET GENE SET DICTIONARIES ---

	import bayesian_gene_set_enrichment

	# ** init gene set dict of dicts **
	gene_set_dict_of_dicts = {}

	# ** gene promoter motif's  **

	# load motif LRT df
	lrt_thresh = 3 / np.log10(np.e) # 2 / np.log10(np.e)
	motif_lrt_path = '%s/Dropbox/genome/mm10/fasta/promoter/up_1000_down_500_use-gene-symbol_motif_gene_log_lrt_mat_df.tsv' % head_folder
	motif_lrt_df = pd.read_table(motif_lrt_path,sep='\t',index_col='gene')
	motif_lrt_df = motif_lrt_df >= lrt_thresh

	# make the motif gene set dict
	motif_gene_set_dict = {}
	for col in motif_lrt_df.columns:
		genes = motif_lrt_df[motif_lrt_df[col]].index
		if len(genes) > 0:
			motif_gene_set_dict[col] = list(genes)
			
	# add to gene_set_dict_of_dicts
	gene_set_dict_of_dicts['jaspar'] = motif_gene_set_dict

			
	# ** GSEA sets **
	gsea_set_names = ['gsea_kegg_pathways_gene_sets','go_biological_process', 'gsea_go_cellular_component', 'gsea_go_molecular_function']
	for gsea_set_name in gsea_set_names:
		
		# path and load
		try:
			gsea_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/%s' % (head_folder,gsea_set_name)
			gsea_dict = bayesian_gene_set_enrichment.parse_gsea_set_file(gsea_set_path)
		except:
			gsea_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/%s.gmt' % (head_folder,gsea_set_name)
			gsea_dict = bayesian_gene_set_enrichment.parse_gsea_set_file(gsea_set_path)
			
		# add to gene_set_dict_of_dicts
		gene_set_dict_of_dicts[gsea_set_name] = gsea_dict


	# ** KEGG_2019_Mouse **
	gsea_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/KEGG_2019_Mouse.txt' % (head_folder)
	gsea_dict = bayesian_gene_set_enrichment.parse_gsea_set_file(gsea_set_path)
	gene_set_dict_of_dicts['KEGG_2019_Mouse'] = gsea_dict


	# ** KEGG_2019_Mouse **
	gsea_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/TRANSFAC_and_JASPAR_PWMs.txt' % (head_folder)
	gsea_dict = bayesian_gene_set_enrichment.parse_gsea_set_file(gsea_set_path)
	gene_set_dict_of_dicts['TRANSFAC_and_JASPAR_PWMs'] = gsea_dict



	# ** SMC phenotypic switching genes **

	# DE
	smc_switching_de_path = '%s/dropbox/athero_data/preprocessed/RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_scale_True_zero-centered_True_scale-per-batch_False_sample-min-disp_3.0_zsgreen-min_disp_3.0_all-min-disp_0.7/desc_results/subject_RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_dims_128_resolution_0.35_tol_0.005_run_0/pseudotime_X_velocity_results/nichenet_pseudotime-pseudotime_2_numbins-11_binsubset-[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_startdebins-[0]_enddebins-[4]/de_genes.txt' % head_folder
	with open(smc_switching_de_path) as file_obj:
		smc_switching_de_genes = list(map(lambda x: x.replace('\n',""), file_obj.readlines()))
		if reg_genes_to_est:
			smc_switching_de_genes = list(filter(lambda x: x in reg_genes_to_est, smc_switching_de_genes))

	# down
	smc_switching_down_path = '%s/dropbox/athero_data/preprocessed/RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_scale_True_zero-centered_True_scale-per-batch_False_sample-min-disp_3.0_zsgreen-min_disp_3.0_all-min-disp_0.7/desc_results/subject_RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_dims_128_resolution_0.35_tol_0.005_run_0/pseudotime_X_velocity_results/nichenet_pseudotime-pseudotime_2_numbins-11_binsubset-[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_startdebins-[0]_enddebins-[4]/down_genes.txt' % head_folder
	with open(smc_switching_down_path) as file_obj:
		smc_switching_down_genes = list(map(lambda x: x.replace('\n',""), file_obj.readlines()))
		if reg_genes_to_est:
			smc_switching_down_genes = list(filter(lambda x: x in reg_genes_to_est, smc_switching_down_genes))

	# up
	smc_switching_up_path = '%s/Dropbox/athero_data/preprocessed/RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_scale_True_zero-centered_True_scale-per-batch_False_sample-min-disp_3.0_zsgreen-min_disp_3.0_all-min-disp_0.7/desc_results/subject_RPS001-RPS002-RPS003-RPS004-RPS007-RPS008-RPS011-RPS012_dims_128_resolution_0.35_tol_0.005_run_0/pseudotime_X_velocity_results/nichenet_pseudotime-pseudotime_2_numbins-11_binsubset-[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_startdebins-[0]_enddebins-[4]/up_genes.txt' % head_folder
	with open(smc_switching_up_path) as file_obj:
		smc_switching_up_genes = list(map(lambda x: x.replace('\n',""), file_obj.readlines()))
		if reg_genes_to_est:
			smc_switching_up_genes = list(filter(lambda x: x in reg_genes_to_est, smc_switching_up_genes))
		
	# add to dict
	gene_set_dict_of_dicts['smc_switching'] = {"de" : smc_switching_de_genes, 
											   'up' : smc_switching_up_genes, 
											   'down' : smc_switching_down_genes}


	# ** SMC switch ATf3 paper **

	# up
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/smc_switch_rna_up_chrom_open.tsv' % (head_folder)
	smc_switching_up_2 = list(pd.read_table(gene_set_path,header=None,sep='\t').iloc[:,0])

	# down
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/smc_switch_rna_down_chrom_close.tsv' % (head_folder)
	smc_switching_down_2 = list(pd.read_table(gene_set_path,header=None,sep='\t').iloc[:,0])

	# add to dict
	gene_set_dict_of_dicts['smc_switching_2'] = { 
											   'up' : smc_switching_up_2, 
											   'down' : smc_switching_down_2}



	# ** Macrophase M1 vs. M2 **
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/macrophage_M1_VS_M2.gmt' % (head_folder)
	gene_set_dict_of_dicts['macrophage_m1_m2']  = bayesian_gene_set_enrichment.parse_gsea_set_file(gene_set_path)



	# ** KLF4 / Oct4 SMC (and non-SMC) targets **

	# KLF SMC ChIP
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/klf4_smc_hfd_targets.csv' % (head_folder)
	klf_smc_targets_1 = list(pd.read_table(gene_set_path,header=None,sep=',').iloc[:,0])

	# KLF non-SMC ChIP
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/klf4_non_smc_hfd_targets.csv' % (head_folder)
	klf_non_smc_targets = list(pd.read_table(gene_set_path,header=None,sep=',').iloc[:,0])

	# KLF SMC ChIP 2
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/klf4_smc_chip.tsv' % (head_folder)
	klf_smc_targets_2 = list(pd.read_table(gene_set_path,header=None,sep='\t').iloc[:,0])

	# Oct4 SMC ChIP
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/oct4_smc_chip.tsv' % (head_folder)
	oct4_smc_targets = list(pd.read_table(gene_set_path,header=None,sep='\t').iloc[:,0])

	# add to dict
	gene_set_dict_of_dicts['klf4_oct4_targets'] ={
		"klf_smc_targets_1" : klf_smc_targets_1,
		'klf_non_smc_targets' : klf_non_smc_targets,
		'klf_smc_targets_2'  : klf_smc_targets_2,
		'oct4_smc_targets' : oct4_smc_targets
	}


	# ** aging genes **
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/aging_mls_all_tissues.csv' % (head_folder)
	df = pd.read_table(gene_set_path,sep=',')
	neg_mls_genes = list(df[(df['Spearman with MLS after BM correction'] < -1 * 0.2) & (df['BH adjusted p.1'] < 0.05)]['Gene'])
	pos_mls_genes = list(df[(df['Spearman with MLS after BM correction'] > 0.2) & (df['BH adjusted p.1'] < 0.05)]['Gene'])
	gene_set_dict_of_dicts['mls_genes'] ={
		"pos_mls_genes" : pos_mls_genes,
		'neg_mls_genes' : neg_mls_genes
	}

	# ** still need to add macrophage circadian young / old **

	# ** CAD markers **
	gene_set_path = '%s/Dropbox/aorta_circadian_data/gene_sets/cad_genes.csv' % (head_folder)
	cad_genes = list(pd.read_table(gene_set_path,header=None,sep=',').iloc[:,0])
	cad_genes = list(map(lambda x: x.lower().capitalize(), cad_genes))
	gene_set_dict_of_dicts['cad_genes'] ={
		"cad" : cad_genes
	}


	return gene_set_dict_of_dicts





# --- violin ----

def make_temporal_violin_plots(sampled_log_props, condit_label, condits, gene_of_interest, gene_names, use_title = True, fileout = None, figsize = (3,1.6)):

	# ** get the gene index **
	if gene_of_interest not in gene_names:
		print("%s not in gene names" % gene_of_interest)
		return 
	gene_index = int(np.where(np.array(gene_names) == gene_of_interest)[0])



	# ** prop distribs over time **

	# make gene df
	# try:
	# 	sampled_log_props = list(map(lambda x: x[:,:,gene_index.item()].unsqueeze(2), sampled_log_props))
	# except:
	sampled_log_props = list(map(lambda x: x[:,:,gene_index].unsqueeze(2), sampled_log_props))
	val_mat = torch.cat(sampled_log_props,dim=2)
	time_mat = np.zeros((sampled_log_props[0].shape[0], 4, len(sampled_log_props)))
	time_mat[:,0,:] = 0
	time_mat[:,1,:] = 6
	time_mat[:,2,:] = 12
	time_mat[:,3,:] = 18
	hue_mat = np.zeros((sampled_log_props[0].shape[0], 4, len(sampled_log_props))).astype(str)
	for i, condit in enumerate(condits):
		hue_mat[:,:,i] = condit
	gene_df = pd.DataFrame()
	gene_df['ZT'] = time_mat.flatten().astype(int)
	gene_df[condit_label] = hue_mat.flatten()
	gene_df['Log10 prop'] = val_mat.flatten().numpy() / np.log(10)


	# viz
	fig, ax = plt.subplots(figsize = figsize,dpi=300)
	sns.despine()
	sns.violinplot(x="ZT", y="Log10 prop", data=gene_df,ax=ax,hue=condit_label)
	plt.xlabel("ZT (hours)")
	plt.ylabel("Log10 proportion")
	if use_title:
		plt.title(gene_of_interest)
	if fileout:
		plt.savefig(fileout,bbox_inches='tight')
		
	return fig, ax





