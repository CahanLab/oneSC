from sklearn.decomposition import PCA 
import umap
import numpy as np 
import pandas as pd
 

def UMAP_embedding_train(train_exp): 
    """Calculate UMAP embedder based on top 9 PCs 

    Args:
        train_exp (pandas.DataFrame): expression profile. Columns are samples and rows are genes. 

    Returns:
        dict: dictionary object needed to apply UMAP 
    """
    training_obs = dict()
    training_obs['feature_genes'] = np.array(train_exp.index)
    train_exp = train_exp.T

    my_PCA = PCA()
    my_PCA.fit(train_exp)

    training_obs['PCA'] = my_PCA

    PCA_features = my_PCA.transform(train_exp)
    if PCA_features.shape[1] < 9: 
        n_PC = PCA_features.shape[1]
    else: 
        n_PC = 9
    PCA_features = pd.DataFrame(PCA_features, index = train_exp.index)
    PCA_features = PCA_features.iloc[:, 0:n_PC]

    my_UMAP = umap.UMAP().fit(PCA_features)
    training_obs['UMAP'] = my_UMAP

    return training_obs

def UMAP_embedding_apply(train_obs, query_exp): 
    feature_genes = train_obs['feature_genes']
    my_PCA = train_obs['PCA']
    my_UMAP = train_obs['UMAP']

    query_exp = query_exp.loc[feature_genes, :]
    query_exp = query_exp.T 
    
    PCA_features = my_PCA.transform(query_exp)
    if PCA_features.shape[1] < 9: 
        n_PC = PCA_features.shape[1]
    else: 
        n_PC = 9
    PCA_features = pd.DataFrame(PCA_features, index = query_exp.index)
    PCA_features = PCA_features.iloc[:, 0:n_PC]

    UMAP_matrix = my_UMAP.transform(PCA_features)
    UMAP_matrix = pd.DataFrame(UMAP_matrix, index = PCA_features.index, columns = ['UMAP_1', 'UMAP_2'])
    return UMAP_matrix

def gaussian_kernel(x, mean, stdv):
    bandwidth = np.linalg.norm(stdv)
    center = x - mean
    guassian_result = np.exp(-np.linalg.norm(center) / (2 * bandwidth))
    return guassian_result

def select_regulators_transition(transition_dict, num_prev = 2): 
    potential_reg_dict = dict()
    
    for temp_lineage in transition_dict.keys(): 
        temp_trans = transition_dict[temp_lineage]
        for i in range(1, temp_trans.shape[1]):
            cur_col = temp_trans.iloc[:, i]
            
            if i - num_prev < 0: 
                prev_index = 0
            else: 
                prev_index = i - num_prev
            prev_col = temp_trans.iloc[:, prev_index:i]
            prev_col = np.abs(prev_col)
            prev_col = prev_col.sum(axis = 1)
            prev_col = prev_col[prev_col != 0]
            cur_col = cur_col[cur_col != 0]
            
            for temp_gene in cur_col.index: 
                if temp_gene in potential_reg_dict.keys(): 
                    potential_reg_dict[temp_gene] = np.unique(np.append(potential_reg_dict[temp_gene], np.array(prev_col.index)))
                    potential_reg_dict[temp_gene] = np.unique(np.append(potential_reg_dict[temp_gene], np.array(cur_col.index)))
                else:
                    potential_reg_dict[temp_gene] = np.unique(np.append(np.array(cur_col.index), np.array(prev_col.index)))
    return potential_reg_dict

def select_regulators_time_series(lineage_time_change_dict, activation_window = 0.25, lag = 0.02): 
    potential_reg_dict = dict()
    for temp_lineage in lineage_time_change_dict.keys(): 
        full_df = lineage_time_change_dict[temp_lineage].copy()
        partial_df = lineage_time_change_dict[temp_lineage].copy()
        partial_df = partial_df.loc[partial_df['PseudoTime'] > lag, :]
        
        for idx in partial_df.index: 
            sub_df = full_df.loc[np.logical_and(full_df['PseudoTime'] >= (partial_df.loc[idx, 'PseudoTime'] - activation_window), full_df['PseudoTime'] < partial_df.loc[idx, 'PseudoTime'] - lag), :]
            temp_gene = partial_df.loc[idx, 'gene']
            if temp_gene in potential_reg_dict.keys(): 
                potential_reg_dict[temp_gene] = np.append(potential_reg_dict[temp_gene], sub_df['gene'])
                potential_reg_dict[temp_gene] = np.unique(potential_reg_dict[temp_gene])
            else:
                potential_reg_dict[temp_gene] = np.unique(sub_df['gene'])
    return potential_reg_dict