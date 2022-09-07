import numpy as np 
import pandas as pd 
from .gene import *
from sklearn.neighbors import KDTree
from scipy.optimize import curve_fit

class network_structure(object):
    def __init__(self, subnet_name):
        self.subnet_name = subnet_name
        self.network_dict = dict()
        
    def train_dummy_grn(self, grn_tab, max_val = 2, min_val = 0.02, m = 7):
        network_dict = dict()
   
        all_genes = np.concatenate((np.unique(grn_tab['TF']), np.unique(grn_tab['TG'])))
        self.TFs = np.unique(grn_tab['TF'])
        self.orphan_genes = np.setdiff1d(grn_tab['TF'], grn_tab['TG'])    
        
        for temp_gene in all_genes: 
            temp_grn = grn_tab.loc[grn_tab['TG'] == temp_gene, :]
            temp_TFs = np.unique(temp_grn['TF'])
            
            gene_obj = gene("normal", "temp_gene")
            gene_obj.add_upstream_genes(temp_TFs)

            temp_combo_dict = dict()
            temp_combo_dict['activation'] = np.array([])
            temp_combo_dict['repression'] = np.array([])
            
            if "+" in np.unique(temp_grn['Type']): 
                activating_grn = temp_grn.loc[temp_grn['Type'] == "+", :]
                temp_combo_dict['activation'] = np.array(activating_grn['TF'])
            
            if "-" in np.unique(temp_grn['Type']): 
                repressing_grn = temp_grn.loc[temp_grn['Type'] == "-", :]
                temp_combo_dict['repression'] = np.array(repressing_grn['TF'])
            
            gene_obj.add_regulation_combo(temp_combo_dict)
            
            gene_obj.add_regulation_coeff(max_val) 
            norm_factors = dict()
            norm_factors['max'] = max_val
            norm_factors['min'] = min_val
            norm_factors['b'] = ((max_val - min_val) / 2) + min_val
            norm_factors['m'] = m

            gene_obj.add_norm_factors(norm_factors)
            gene_obj.add_max_exp(max_val)
            network_dict[temp_gene] = gene_obj
        self.network_dict = network_dict

    def train_grn(self, grn_tab, exp_tab, samp_tab, cluster_col = "cell_type"):
        network_dict = dict()
   
        all_genes = np.concatenate((np.unique(grn_tab['TF']), np.unique(grn_tab['TG'])))
        self.TFs = np.unique(grn_tab['TF'])
        self.orphan_genes = np.setdiff1d(grn_tab['TF'], grn_tab['TG'])    
        
        for temp_gene in all_genes: 
            temp_grn = grn_tab.loc[grn_tab['TG'] == temp_gene, :]
            temp_TFs = np.unique(temp_grn['TF'])
            
            gene_obj = gene("normal", "temp_gene")
            gene_obj.add_upstream_genes(temp_TFs)

            temp_combo_dict = dict()
            temp_combo_dict['activation'] = np.array([])
            temp_combo_dict['repression'] = np.array([])
            
            if "+" in np.unique(temp_grn['Type']): 
                activating_grn = temp_grn.loc[temp_grn['Type'] == "+", :]
                temp_combo_dict['activation'] = np.array(activating_grn['TF'])
            
            if "-" in np.unique(temp_grn['Type']): 
                repressing_grn = temp_grn.loc[temp_grn['Type'] == "-", :]
                temp_combo_dict['repression'] = np.array(repressing_grn['TF'])
            
            gene_obj.add_regulation_combo(temp_combo_dict)

            cluster_exp_list = []
            for temp_cluster in np.unique(samp_tab[cluster_col]):
                temp_samp_tab = samp_tab.loc[samp_tab[cluster_col] == temp_cluster, :]
                cluster_val = exp_tab.loc[temp_gene, temp_samp_tab.index].mean()
                cluster_exp_list.append(cluster_val)

            gene_obj.add_regulation_coeff(np.max(cluster_exp_list))
            
            #dummy_logistic = LogisticRegression().fit(np.matrix([0, 0, 1, 1]).T, [0, 0, 1, 1])
            norm_factors = dict()
            norm_factors['max'] = np.max(cluster_exp_list)
            norm_factors['min'] = np.min(cluster_exp_list)
            norm_factors['b'] = ((np.max(cluster_exp_list) - np.min(cluster_exp_list)) / 2) + np.min(cluster_exp_list)
            
            for m in list(range(0, 60)): 
                scaled_max = 1 / (1 + np.exp(-m * (norm_factors['max'] - norm_factors['b'])))
                scaled_min = 1 / (1 + np.exp(-m * (norm_factors['min'] - norm_factors['b'])))
                if scaled_max > 0.99 and scaled_min < 0.01: 
                    break 
            norm_factors['m'] = m 

            gene_obj.add_max_exp(np.max(cluster_exp_list)) # just a way to limit the max 
            gene_obj.add_norm_factors(norm_factors)
            
            network_dict[temp_gene] = gene_obj
                
        self.network_dict = network_dict