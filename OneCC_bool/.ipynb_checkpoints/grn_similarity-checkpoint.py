import numpy as np 
import pandas as pd 

def gaussian_kernel(x, mean, stdv):
    bandwidth = np.linalg.norm(stdv)
    center = x - mean
    guassian_result = np.exp(-np.linalg.norm(center) / (2 * bandwidth))
    return guassian_result

def calc_grn(sim_exp_path): 
    sim_exp = pd.read_csv(sim_exp_path, index_col=0)
    grn_list = []
    for samp_id in sim_exp.columns: 
        grn_list.append(gaussian_kernel(sim_exp.loc[:, samp_id], temp_mean, temp_sd))
    return grn_list

