import numpy as np 
import pandas as pd
import scipy.stats
from sklearn.mixture import GaussianMixture

# this function is to find the threshold of the genes based on the data observation 
# this could obviously be a little bit better 
def find_threshold_vector(exp_df, samp_st, cluster_col = "cluster"): 
    cluster_exp = pd.DataFrame()
    for temp_cluster in np.unique(samp_st[cluster_col]):
        temp_st = samp_st.loc[samp_st[cluster_col] == temp_cluster, :]
        temp_exp = exp_df.loc[:, temp_st.index]
        cluster_exp[temp_cluster] = temp_exp.mean (axis = 1)
    return (cluster_exp.max(axis = 1) - exp_df.min(axis = 1)) / 2

# this is the function to booleanizering expressions across the state 
def find_boolean_across_time(exp_df, samp_st, cluster_col, trajectory_list, bool_thresholds):
    mean_bool_df = pd.DataFrame()
    for temp_cluster in trajectory_list:
        temp_st = samp_st.loc[samp_st[cluster_col] == temp_cluster, :]
        temp_exp = exp_df.loc[:, temp_st.index]
        mean_bool_df[temp_cluster] = temp_exp.mean(axis = 1) > bool_thresholds
    mean_bool_df.columns = list(range(0, len(trajectory_list)))
    return mean_bool_df * 1

# find the diff vectors across the time series 
def find_diff_boolean(mean_bool_df): 
    diff_bool_df = pd.DataFrame()
    for i in list(range(0, mean_bool_df.shape[1])):
        if i == 0:
            temp_diff_bool = mean_bool_df[i]
            temp_diff_bool = temp_diff_bool.replace(0, -1)
            diff_bool_df[i] = temp_diff_bool
        else:
            # if the difference is 0 then skip 
            temp_mean_diff = mean_bool_df[i] - mean_bool_df[i - 1]
            if temp_mean_diff.eq(0).all() == False:
                diff_bool_df[i] = temp_mean_diff
    
    diff_bool_df.columns = list(range(0, diff_bool_df.shape[1]))
    return diff_bool_df

# suggested some edge using average pearson correlation across different lineage 
def edge_suggestion(TG, TFs, train_exp, train_st, trajectory_cells_dict, cluster_col = 'cluster_label'): 
    
    corr_list = list()
    raw_corr_list = list()

    for temp_TF in TFs.index: 
        temp_corr = list()
        temp_raw_corr = list()

        for trajectory_state in trajectory_cells_dict.keys():
            trajectory_clusters = trajectory_cells_dict[trajectory_state]
            sub_train_st = train_st.loc[train_st[cluster_col].isin(trajectory_clusters), :]
            sub_train_exp = train_exp.loc[:, sub_train_st.index]

            TG_exp = sub_train_exp.loc[list(TG.index), :]
            TG_exp = np.array(TG_exp.iloc[0, :])

            TF_exp = np.array(sub_train_exp.loc[temp_TF, :])

            corr_result = TFs[temp_TF] * TG[0] * scipy.stats.pearsonr(TF_exp, TG_exp)[0]
            if corr_result < 0: 
                continue
            else: 
                temp_corr.append(corr_result)
                temp_raw_corr.append(scipy.stats.pearsonr(TF_exp, TG_exp)[0])
        if len(temp_corr) == 0:
            corr_list.append(-1)
            raw_corr_list.append(-1)      
        else:
            corr_list.append(np.mean(temp_corr))
            raw_corr_list.append(np.mean(temp_raw_corr))
    stats_tab = pd.DataFrame()
    stats_tab['TF'] = TFs.index
    stats_tab['correlation'] = corr_list
    stats_tab['raw_corr'] = raw_corr_list

    stats_tab = stats_tab.sort_values("correlation", axis = 0, ascending=False)
    return stats_tab 

def compile_lineage_sampTab(train_exp, samp_tab, pt_col): 
    
    temp_pseudo_time = samp_tab[pt_col]
    temp_pseudo_time = temp_pseudo_time.dropna()
    
    temp_train_exp = train_exp.loc[:, temp_pseudo_time.index]
    sub_train_gmm = temp_train_exp.copy()
    sub_train_gmm = sub_train_gmm.T
    sub_train_gmm['pseudoTime'] = temp_pseudo_time
    
    all_ks = list(range(2, 20))
    BIC_scores = list()
    for temp_k in all_ks: 
        gm = GaussianMixture(n_components=temp_k, random_state=0).fit(sub_train_gmm)
        cluster_label = gm.predict(sub_train_gmm)
        BIC_scores.append(gm.bic(sub_train_gmm))
        
    selected_k = all_ks[np.argmin(BIC_scores)]
    
    gm = GaussianMixture(n_components=selected_k, random_state=0).fit(sub_train_gmm)
    cluster_label = gm.predict(sub_train_gmm)
    cluster_label = [str(x) + "_cluster" for x in cluster_label]
    
    return_sampTab = pd.DataFrame()
    return_sampTab.index = sub_train_gmm.index
    return_sampTab['cluster_label'] = cluster_label 
    return_sampTab['pseudoTime'] = temp_pseudo_time
    
    temp_clusters = np.unique(return_sampTab['cluster_label'])
    avg_pseudoTime = list()

    for temp_cluster in temp_clusters: 
        cluster_temp_st = return_sampTab.loc[return_sampTab['cluster_label'] == temp_cluster, :]
        cluster_temp_pseudotime = cluster_temp_st['pseudoTime']
        avg_pseudoTime.append(np.mean(cluster_temp_pseudotime))

    ordered_clusters = temp_clusters[np.argsort(avg_pseudoTime)]

    return [return_sampTab, ordered_clusters]

# this is to run the wrapper of trimming using lineage 
def lineage_trimming(orig_grn, train_exp, train_st, trajectory_cells_dict, bool_thresholds, pt_col = 'pseudoTime', cluster_col = 'cluster_label'):
    
    # calculate the threshold according to the training data 
    # bool_thresholds = find_threshold_vector(train_exp, train_st, cluster_col)

    # reindex the dataframe to clarity 
    orig_grn.index = list(range(0, orig_grn.shape[0]))

    refined_grn_dict = dict()
    for end_state in trajectory_cells_dict.keys():
        trajectory_clusters = trajectory_cells_dict[end_state]
        
        # shoud still take in clusters 
        sub_train_st = train_st.loc[train_st[cluster_col].isin(trajectory_clusters), :]
        sub_train_exp = train_exp.loc[:, sub_train_st.index]
        
        mean_bool_df = find_boolean_across_time(sub_train_exp, sub_train_st, cluster_col, trajectory_clusters, bool_thresholds)
        mean_diff_bool = find_diff_boolean(mean_bool_df)
        # assign the grn 
        initial_grn = orig_grn.copy()

        running_list = list(range(1, mean_diff_bool.shape[1]))
        
        for run_index in running_list:
            current_diff = mean_diff_bool[run_index]
            current_diff = current_diff[current_diff != 0]

            prev_diff = mean_diff_bool[run_index - 1]
            prev_diff = prev_diff[prev_diff != 0]

            # remove the genes that changed on step back 
            prev_diff = prev_diff[np.setdiff1d(np.array(prev_diff.index), np.array(current_diff.index))]

            # if the the prev difference is 0
            # look at the genes in the current network 
            temp_run_index =  run_index - 1
            while prev_diff.shape[0] == 0:
                prev_diff = mean_diff_bool[temp_run_index - 1]
                prev_diff = prev_diff[prev_diff != 0]
                prev_diff = prev_diff[np.setdiff1d(np.array(prev_diff.index), np.array(current_diff.index))]

                if temp_run_index > 1:
                    temp_run_index = temp_run_index - 1
                if temp_run_index == 1:
                    break 

            for temp_TG in current_diff.index:
                temp_grn = initial_grn.loc[initial_grn['TG'] == temp_TG, :]

                # if there are edges that are not supposed to be there 
                # remove edges to nodes that are not in the prev diff column 
                bad_index = temp_grn.loc[temp_grn['TF'].isin(list(prev_diff.index)) == False, :].index
                bad_index = list(bad_index)

                # Get the TF that should be 
                pos_TFs = prev_diff[prev_diff == current_diff[temp_TG]].index
                neg_TFs = prev_diff[prev_diff != current_diff[temp_TG]].index

                # negative genes cannot have "+"
                pos_og_grn = temp_grn.loc[temp_grn['Type'] == "+", :]
                bad_index = bad_index + list(pos_og_grn.loc[pos_og_grn['TF'].isin(neg_TFs), :].index)

                # positive genes cannot have "-"
                neg_og_grn = temp_grn.loc[temp_grn['Type'] == '-', :]
                bad_index = bad_index + list(neg_og_grn.loc[neg_og_grn['TF'].isin(pos_TFs), :].index)

                bad_index = np.unique(bad_index)

                # drop all the bad edges from the original and temp grn 
                initial_grn = initial_grn.drop(bad_index)
                temp_grn = temp_grn.drop(bad_index)

                if any(temp_grn['TF'].isin(prev_diff.index)) == True: 
                    continue
                else:
                    TG_series = pd.Series(current_diff.loc[temp_TG], [temp_TG])
                    
                    edge_stats = edge_suggestion(TG_series, prev_diff, train_exp, train_st, trajectory_cells_dict, cluster_col)

                    new_edge = pd.DataFrame(None, index = [str(edge_stats.iloc[0, 0] + "_" + temp_TG + "_new")], columns = ["TF", "TG", "Type"])
                    new_edge.iloc[0, 0] = edge_stats.iloc[0, 0]
                    new_edge.iloc[0, 1] = temp_TG
                    
                    # if the sign of the correlation is negative then 
                    # skip because it shouldn't happen 
                    if edge_stats.iloc[0, 1] < 0: 
                        continue
                    if edge_stats.iloc[0, 2] > 0: 
                        new_edge.iloc[0, 2] = "+"
                    else:
                        new_edge.iloc[0, 2] = "-"
                    initial_grn = pd.concat([initial_grn, new_edge])
                
        refined_grn_dict[end_state] = initial_grn
    
    final_refined_df = pd.DataFrame()
    for end_state in refined_grn_dict.keys():
        final_refined_df = pd.concat([final_refined_df, refined_grn_dict[end_state]])
    final_refined_df = final_refined_df.drop_duplicates()
    return final_refined_df







