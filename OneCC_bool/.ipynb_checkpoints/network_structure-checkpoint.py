import numpy as np 
import pandas as pd 
from .gene import *
from sklearn.neighbors import KDTree
from scipy.optimize import curve_fit

class network_structure(object):
    def __init__(self, subnet_name):
        self.subnet_name = subnet_name
        self.network_dict = dict()
        
    def train_dummy_grn(self, grn_tab):
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
            
            gene_obj.add_regulation_coeff(2)
            norm_factors = dict()
            norm_factors['max'] = 1.5
            norm_factors['min'] = 0.02
            
            linear_parameters = dict()
            linear_parameters['m'] = 1.5
            linear_parameters['b'] = 0
            gene_obj.add_linear_parameters(linear_parameters)

            gene_obj.add_norm_factors(norm_factors)
            gene_obj.add_max_exp(2)
            network_dict[temp_gene] = gene_obj
        self.network_dict = network_dict

    def find_neighbors_pseudo(self, expDat, TG, num_pseudo = 50, n_neighbors = 10):
        new_order_column = expDat.loc[TG, :].sort_values(ascending = False).index
        expDat_ordered = expDat.loc[:, new_order_column].copy()

        # sample an equal spaced 
        col_list = list(range(0, expDat_ordered.shape[1], int(expDat_ordered.shape[1]/num_pseudo)))
        expDat_ordered = expDat_ordered.iloc[:, col_list]

        kdt = KDTree(expDat_ordered.T, leaf_size=30, metric='euclidean')
        pseudo_neighbors = kdt.query(expDat_ordered.T, k = n_neighbors, return_distance=False)

        pseudo_df = pd.DataFrame(np.zeros((len(expDat_ordered.index), len(pseudo_neighbors))), index = expDat_ordered.index)
        
        for pseudo_index in range(0, len(pseudo_neighbors)):
            pseudo_neighbor = pseudo_neighbors[pseudo_index]
            temp_exp = expDat_ordered.iloc[:, pseudo_neighbor]
            avg_exp = temp_exp.mean(axis = 1)

            pseudo_df.iloc[:, pseudo_index] = avg_exp
        return pseudo_df
    
    def calc_scale_expression(self, raw_exp, TF_norm_factors):
        #TF_max = TF_norm_factors['max'] * 10 
        TF_max = TF_norm_factors['max']
        #TF_min = TF_norm_factors['min'] * 10 
        TF_min = TF_norm_factors['min']

        scaled_exp = (raw_exp - TF_min) / (TF_max - TF_min)
        scaled_exp = np.array(scaled_exp)
        
        for temp_index in list(range(0, len(scaled_exp))): 
            if scaled_exp[temp_index] < 0: 
                scaled_exp[temp_index] = 0
                
            if scaled_exp[temp_index] > 1: 
                scaled_exp[temp_index] = 1
  
        return scaled_exp 

    def train_grn(self, grn_tab, exp_tab, samp_tab, cluster_col = "cell_type", num_pseudo = 50, n_neighbors = 10, random_seed = 123):
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
            norm_factors['max'] = np.max(cluster_exp_list) * 0.8
            norm_factors['min'] = np.min(cluster_exp_list)
            
            gene_obj.add_max_exp(np.max(cluster_exp_list)) # just a way to limit the max 
            gene_obj.add_norm_factors(norm_factors)
            
            linear_parameters = dict()
            linear_parameters['m'] = 1
            linear_parameters['b'] = 0
            gene_obj.add_linear_parameters(linear_parameters)
            
            network_dict[temp_gene] = gene_obj
                
        self.network_dict = network_dict