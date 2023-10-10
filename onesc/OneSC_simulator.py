import numpy as np 
import pandas as pd 
from .gene import *

class OneSC_simulator(object): 
    def __init__(self): 
        self.networks_compilation = dict()
        self.sim_exp = None
    def add_network_compilation(self, subnet_name, subnet_obj): 
        self.networks_compilation[subnet_name] = subnet_obj
    
    def calc_scale_expression(self, raw_exp, TF_norm_factors):
        TF_mid = TF_norm_factors['b']
        TF_scale = TF_norm_factors['m']

        scaled_exp = 1 / (1 + np.exp(-TF_scale * (raw_exp - TF_mid)))
        if scaled_exp < 0: 
            scaled_exp = 0.001
        if scaled_exp > 1: 
            scaled_exp = 1.001
        return scaled_exp 

    def calc_production_rate(self, prev_exp, TG, network_model, norm_dict):
        TG_gene_obj = network_model.network_dict[TG]
        reg_coefficient = TG_gene_obj.regulation_coeff
        
        def calc_activation_prob(norm_dict, upTFs): 
            if len(upTFs) == 0: 
                return 1
            elif len(upTFs) == 1: 
                return norm_dict[upTFs[0]]
            else: 
                x1 = 0 
                x2 = 0
                for TF in upTFs: 
                    if x1 == 0: 
                        x1 = norm_dict[TF]
                    elif x2 == 0: 
                        x2 = norm_dict[TF]
                        x1 = 1 - ((1 - x1) * (1 - x2))
                        x2 = 0
                return x1
        
        def calc_repression_prob(norm_dict, downTFs): 
            if len(downTFs) == 0: 
                return 1
            elif len(downTFs) == 1: 
                return 1 - norm_dict[downTFs[0]]
            else:
                total_repression = 1
                for TF in downTFs: 
                    total_repression = total_repression * (1 - norm_dict[TF])
                return total_repression
        
        activation_prob = calc_activation_prob(norm_dict, TG_gene_obj.regulation_combo["activation"])
        repression_prob = calc_repression_prob(norm_dict, TG_gene_obj.regulation_combo["repression"])
        
        total_prob = activation_prob * repression_prob 
        if total_prob > 1: 
            total_prob = 1
        if total_prob < 0:
            total_prob = 0       

        return reg_coefficient * total_prob
            
    def simulate_exp(self, initial_exp_dict, initial_subnet, perturb_dict = dict(), decay_rate = 0.1, num_sim = 1000, t_interval = 0.01, noise_amp = 2, stochasticity = True, random_seed = 0):
        np.random.seed(random_seed)
        
        time_series_col = list(range(0, num_sim))
        time_df = pd.DataFrame(data = None, index = initial_exp_dict.keys(), columns = time_series_col)

        network_model = self.networks_compilation[initial_subnet].network_dict

        for time_point in time_df.columns:
            if time_point == 0: 
                for gene_name in initial_exp_dict.keys(): 
                    time_df.loc[gene_name, time_point] = initial_exp_dict[gene_name]
            else: 
                prev_exp = time_df.loc[:, (time_point - 1)]
                time_df.loc[:, time_point] = prev_exp
                
                norm_dict = dict()
                for TF in network_model.keys():
                    TF_norm_factors = network_model[TF].norm_factors
                    scaled_exp = self.calc_scale_expression(prev_exp[TF], TF_norm_factors)
                    norm_dict[TF] = scaled_exp 

                for gene_name in network_model.keys():
                    gene_production = self.calc_production_rate(prev_exp, gene_name, self.networks_compilation[initial_subnet], norm_dict)
                    gene_production = gene_production * decay_rate

                    if gene_name in perturb_dict.keys(): 
                        perturb_production = perturb_dict[gene_name] * decay_rate
                        gene_production = gene_production + perturb_production 

                    if stochasticity == True: 
                        production_signal = (prev_exp[gene_name] - network_model[gene_name].norm_factors['min'])
                        if production_signal < 0: 
                            production_signal = prev_exp[gene_name]
                        noise_p = np.random.normal(0, t_interval) * (production_signal ** 0.5)
                        noise_term = noise_p * noise_amp
                    else: 
                        noise_term = 0 
                    
                    deterministic_exp = prev_exp[gene_name] + ((gene_production - (decay_rate * prev_exp[gene_name])) * t_interval)
                    if deterministic_exp > network_model[gene_name].max_exp:
                        deterministic_exp = network_model[gene_name].max_exp
                    
                    gene_exp_t = deterministic_exp + noise_term 
                    
                    if gene_exp_t < network_model[gene_name].norm_factors['min']:
                        gene_exp_t = network_model[gene_name].norm_factors['min']
                        
                    time_df.loc[gene_name, time_point] = np.round(gene_exp_t, 5) # at most have 5 decimal points 

            # check and terminate the simulation if steady state has been reached 
            if stochasticity == False: 
                if time_df.iloc[:, (time_point)].equals(time_df.iloc[:, (time_point - 1)]): 
                    break 

        time_df = time_df.iloc[:, 0:(time_point + 1)]
        self.sim_exp = time_df