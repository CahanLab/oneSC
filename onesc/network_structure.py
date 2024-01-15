import numpy as np 
import pandas as pd 
from .gene import *

class network_structure(object):
    """A network structure that organizes the upstream regulators and logistic regression models for each gene. 
    """
    def __init__(self):
        """Constructor for a network_structure object 
        """
        self.network_dict = dict()
        
    def fit_grn(self, grn_tab, max_val = 2, min_val = 0.02, m = 7):
        """Fit a systems of equations modelling the transcription regulations governed by a inferred gene regulatory network.
           Set the parameters of logistic function that model the percent activity (for transcription factors) from expression values. 

        Args:
            grn_tab (pd.DataFrame): A pandas dataframe of inferred GRN. TF column represents the regulator, TG column represents target genes and Type column represents the type of regulation (+ or -). 
            max_val (int, optional): The maximum simulated expression value. Defaults to 2.
            min_val (float, optional): The minimum simulated expression value. Defaults to 0.02.
            m (int, optional): The logistic growth rate or steepness of the curve. Defaults to 7.
        """
        network_dict = dict()
   
        all_genes = np.concatenate((np.unique(grn_tab['TF']), np.unique(grn_tab['TG'])))
        self.TFs = np.unique(grn_tab['TF'])
        self.orphan_genes = np.setdiff1d(grn_tab['TF'], grn_tab['TG'])    
        
        for temp_gene in all_genes: 
            temp_grn = grn_tab.loc[grn_tab['TG'] == temp_gene, :]
            temp_TFs = np.unique(temp_grn['TF'])
            
            gene_obj = gene()
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

