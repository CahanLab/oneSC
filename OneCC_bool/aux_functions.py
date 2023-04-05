from sklearn.decomposition import PCA 
import umap
import numpy as np 
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests
from pygam import GAM, s,l
import scanpy as sc

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

def rank_potential_regulators(train_exp):
    correlation_matrix = np.corrcoef(train_exp)
    regulator_dict = dict()
    for temp_index in range(0, len(train_exp.index)):
        temp_corr = correlation_matrix[temp_index]
        ordered_regulators = train_exp.index[np.argsort(-np.abs(temp_corr))]
        ordered_regulators = np.array(ordered_regulators)
        regulator_dict[train_exp.index[temp_index]] = ordered_regulators
    return regulator_dict

def gamFit(expMat,genes,celltime):
    genes2 = (set(genes) & set(expMat.index))
    def fit_gam_per_gene(input_data):
        z = pd.DataFrame()
        z["z"] = input_data.values
        z["t"] = celltime.values
        z.index = expMat.columns
        X = celltime.values.reshape((celltime.shape[0],1))
        y = z["z"].values
        gam = GAM(l(0)).fit(X,y)
        p = gam.statistics_['p_values'][0]
        return p
    ans = expMat.loc[genes2][celltime.index].apply(fit_gam_per_gene,axis=1)
    return ans

def suggest_dynamic_genes(exp_tab, samp_tab, cluster_col):
    average_df = pd.DataFrame(data = None, index = exp_tab.index, columns = np.unique(samp_tab[cluster_col]))
    for gene in average_df.index: 
        for cell_type in np.unique(samp_tab[cluster_col]): 
            temp_samp_tab = samp_tab.loc[samp_tab[cluster_col] == cell_type, :]
            temp_exp = exp_tab.loc[gene, temp_samp_tab.index]
            average_df.loc[gene, cell_type] = np.mean(temp_exp)

    average_df['min_exp'] = None
    average_df['max_exp'] = None 
    average_df['log2_change'] = None

    for gene in average_df.index: 
        min_exp = np.min(average_df.loc[gene, :].dropna())
        max_exp = np.max(average_df.loc[gene, :].dropna())
        
        average_df.loc[gene, 'min_exp'] = min_exp
        average_df.loc[gene, 'max_exp'] = max_exp 
    
    average_df.loc[average_df['min_exp'] == 0, 'min_exp'] = 0.001
    average_df.loc[average_df['max_exp'] == 0, 'max_exp'] = 0.001

    fold_change = np.array(average_df['max_exp'] / average_df['min_exp']).astype(float)
    average_df.loc[:, 'log2_change'] = np.log2(fold_change)

    return average_df

def suggest_dynamic_TFs(exp_tab, samp_tab, tf_list, cluster_col, n_top_genes = 2000, adj_p_cutoff = 0.05, logfold_cutoff = 2, pct_exp_cutoff = 0.1):
    temp_adata = sc.AnnData(exp_tab.T)
    temp_adata.obs = samp_tab
    sc.pp.highly_variable_genes(temp_adata, n_top_genes = n_top_genes,flavor = 'seurat')
    temp_adata = temp_adata[:, temp_adata.var.highly_variable]
    i_TFs = np.intersect1d(np.array(temp_adata.var.index), tf_list)
    exp_tab = exp_tab.loc[i_TFs, :].copy()
    temp_adata = temp_adata[:, i_TFs].copy()
    average_df = pd.DataFrame(data = None, index = exp_tab.index, columns = np.unique(samp_tab[cluster_col]))
    pval_list = list()
    logfold_list = list()
    percent_list = list()
    for gene in average_df.index: 
        for cell_type in np.unique(samp_tab[cluster_col]): 
            temp_samp_tab = samp_tab.loc[samp_tab[cluster_col] == cell_type, :]
            temp_exp = exp_tab.loc[gene, temp_samp_tab.index]
            average_df.loc[gene, cell_type] = np.mean(temp_exp)
        min_cat = average_df.columns[np.where(average_df.loc[gene, :] == np.min(average_df.loc[gene, :]))[0]][0]
        max_cat = average_df.columns[np.where(average_df.loc[gene, :] == np.max(average_df.loc[gene, :]))[0]][0]
        # write a function that performs wilcoxon rank sum test 
        sub_temp_adata = temp_adata[temp_adata.obs[cluster_col].isin([min_cat, max_cat]), gene].copy()
        sc.tl.rank_genes_groups(sub_temp_adata, cluster_col, method='wilcoxon', pts = True)
        test_results = sc.get.rank_genes_groups_df(sub_temp_adata, group = max_cat)
        pval_list.append(test_results.loc[0, 'pvals'])
        logfold_list.append(test_results.loc[0, 'logfoldchanges'])
        percent_list.append(test_results.loc[0, 'pct_nz_group'])
    average_df['p_val'] = pval_list
    average_df['adj_p'] = multipletests(pval_list, method = 'fdr_bh')[1]
    average_df['logfold'] = logfold_list
    average_df['pct_exp'] = percent_list
    average_df = average_df.loc[average_df['adj_p'] < adj_p_cutoff, :].copy()
    average_df = average_df.loc[average_df['logfold'] > logfold_cutoff, :].copy()
    average_df = average_df.loc[average_df['pct_exp'] > pct_exp_cutoff, :].copy()
    return average_df