import numpy as np 
import pandas as pd
import itertools
import networkx as nx 


def find_gene_change_trajectory(train_exp, train_st, trajectory_cells_dict, cluster_col, pt_col, bool_thresholds, pseudoTime_bin, smooth_style = 'mean'):
    trajectory_time_change_dict = dict()
    for temp_trajectory in trajectory_cells_dict.keys(): 
        activation_time = find_genes_time_change(train_st, train_exp, trajectory_cells_dict, cluster_col, pt_col, bool_thresholds, pseudoTime_bin, temp_trajectory)
        trajectory_time_change_dict[temp_trajectory] = activation_time
    return trajectory_time_change_dict

def find_genes_time_change(train_st, train_exp, trajectory_cluster_dict, cluster_col, pt_col, vector_thresh, pseudoTime_bin, trajectory_name, smooth_style = 'mean'):
    target_clusters = trajectory_cluster_dict[trajectory_name]
    sub_train_st = train_st.loc[train_st[cluster_col].isin(target_clusters), :]
    sub_train_st = sub_train_st.sort_values(pt_col)
    sub_train_exp = train_exp.loc[:, sub_train_st.index]

    activation_time_df = pd.DataFrame()
    for temp_gene in sub_train_exp.index: 
        time_series = pd.DataFrame()
        time_series['expression'] = sub_train_exp.loc[temp_gene, :]
        time_series['PseudoTime'] = sub_train_st[pt_col]

        smoothed_series = bin_smooth(time_series, pseudoTime_bin, smooth_style = smooth_style, spline_ME = 0.1)
        smoothed_series['boolean'] = smoothed_series['expression'] > vector_thresh[temp_gene]

        temp_change_df = find_transition_time(smoothed_series, temp_gene)
        activation_time_df = pd.concat([activation_time_df, temp_change_df])
    
    activation_time_df = activation_time_df.sort_values("PseudoTime")
    activation_time_df.index = list(range(0, activation_time_df.shape[0]))
    return activation_time_df

# this function is to find time points across the dataset 
def find_transition_time(smoothed_series, temp_gene):
    current_status = smoothed_series.loc[0, 'boolean']
    transition_df = pd.DataFrame(index = ['PseudoTime', 'gene', 'type'])
    
    lag = 5
    # if the initial point is activated 
    if current_status == True: 
        transition_df = pd.concat([transition_df, pd.Series({'PseudoTime': 0, 'gene': temp_gene, 'type': '+'})], axis = 1)
    elif current_status == False: 
        transition_df = pd.concat([transition_df, pd.Series({'PseudoTime': 0, 'gene': temp_gene, 'type': '-'})], axis = 1)

    for index in smoothed_series.index[0:len(smoothed_series.index)-lag]: 
        if smoothed_series.loc[index, 'boolean'] == current_status:
            continue
        else: 
            if np.sum(smoothed_series.loc[index:index+lag, 'boolean']) == 0 or np.sum(smoothed_series.loc[index:index+lag, 'boolean']) == len(smoothed_series.loc[index:index+lag, 'boolean']):
                if current_status == False and smoothed_series.loc[index, 'boolean'] == True: 
                    reg_type = "+"
                elif current_status == True and smoothed_series.loc[index, 'boolean'] == False: 
                    reg_type = "-"
            
                transition_df = pd.concat([transition_df, pd.Series({'PseudoTime': smoothed_series.loc[index, 'PseudoTime'], 'gene': temp_gene, 'type': reg_type})], axis = 1)
                current_status = smoothed_series.loc[index, 'boolean']
    
    transition_df = transition_df.T
    transition_df.index = list(range(0, transition_df.shape[0]))
  
    return transition_df

def bin_smooth(time_series, pseudoTime_bin, smooth_style = "mean", spline_ME = 0.1):
    curr_time =  np.min(time_series['PseudoTime'])
    time_list = list()
    smoothed_exp = list()
    stand_dev = list()

    while curr_time < np.max(time_series['PseudoTime']): 
        temp_time_series = time_series.loc[np.logical_and(time_series['PseudoTime'] >= curr_time, time_series['PseudoTime'] < curr_time + pseudoTime_bin), :]
        
        # in the case of 0 expression just move on to the next time frame and move on 
        if temp_time_series.shape[0] == 0:
            curr_time = curr_time + pseudoTime_bin
            continue

        time_list.append(curr_time)
        if smooth_style == 'mean':
            smoothed_exp.append(temp_time_series['expression'].mean())
        elif smooth_style == "median":
            smoothed_exp.append(temp_time_series['expression'].median())

        curr_time = curr_time + pseudoTime_bin
        stand_dev.append(temp_time_series['expression'].std())

    # spline_w = np.divide(1, stand_dev)

    smoothed_data = pd.DataFrame()
    smoothed_data['PseudoTime'] = time_list
    smoothed_data['expression'] = smoothed_exp

    #spline_s = smoothed_data.shape[0] * (spline_ME ** 2)
    #spline_xy = UnivariateSpline(smoothed_data['PseudoTime'],smoothed_data['expression'], s = spline_s)
    #moothed_data['splined_exp'] = spline_xy(smoothed_data['PseudoTime'])
    return smoothed_data 

# this function is to find the threshold of the genes based on the data observation 
# this could obviously be a little bit better 
def find_threshold_vector(exp_df, samp_st, cluster_col = "cluster", cutoff_percentage = 0.4): 
    cluster_exp = pd.DataFrame()
    for temp_cluster in np.unique(samp_st[cluster_col]):
        temp_st = samp_st.loc[samp_st[cluster_col] == temp_cluster, :]
        temp_exp = exp_df.loc[:, temp_st.index]
        cluster_exp[temp_cluster] = temp_exp.mean(axis = 1)
    return ((cluster_exp.max(axis = 1) - cluster_exp.min(axis = 1)) * cutoff_percentage + cluster_exp.min(axis = 1))

def construct_cluster_network(train_exp, sampTab, initial_clusters, terminal_clusters, cluster_col = "cluster_id", pseudo_col = "pseudotime"):
    pt_list = list()
    cluster_list = list()
    mean_exp = pd.DataFrame()
    for temp_cluster in np.unique(sampTab[cluster_col]):
        temp_sampTab = sampTab.loc[sampTab[cluster_col] == temp_cluster, :]
        temp_train_exp = train_exp.loc[:, temp_sampTab.index]
        mean_exp[temp_cluster] = temp_train_exp.mean(axis = 1)
        cluster_list.append(temp_cluster)
        pt_list.append(temp_sampTab[pseudo_col].mean())

    mean_pt = pd.DataFrame()
    mean_pt['cluster'] = cluster_list 
    mean_pt['pt'] = pt_list
    mean_pt = mean_pt.sort_values("pt")
    mean_exp = mean_exp.loc[:, mean_pt['cluster']]

    # calculate the euclidean distance between distance 
    distance_df = pd.DataFrame() 
    for i in range(mean_exp.shape[1]):
        for j in range(i + 1, mean_exp.shape[1]):
            temp_dict = {"starting": mean_exp.columns[i], "ending": mean_exp.columns[j], "distance": np.linalg.norm(mean_exp.iloc[:, j] - mean_exp.iloc[:, i])}
            distance_df = pd.concat([distance_df, pd.DataFrame([temp_dict])], ignore_index = True)
    distance_df['norm_dist'] = (distance_df['distance'] - np.min(distance_df['distance'])) / (np.max(distance_df['distance']) - np.min(distance_df['distance']))
    distance_df['starting_pt'] = None
    mean_pt.index = mean_pt['cluster']
    for cluster in np.unique(distance_df['starting']):
        distance_df.loc[distance_df['starting'] == cluster, 'starting_pt'] = mean_pt.loc[cluster, 'pt']
    distance_df['combined_score'] = distance_df['norm_dist'] + (distance_df['starting_pt'] / 2)

    # add in the weight of the time component. The earlier the better 
    my_G = nx.DiGraph()
    for cluster_name in mean_exp.columns:
        my_G.add_node(cluster_name)
        # if the cluster name is the inital 
        if cluster_name not in initial_clusters:
            # no need to check incoming 
            incoming_nodes = [x[0] for x in my_G.in_edges(cluster_name)]
            if len(incoming_nodes) == 0: 
                try: 
                    temp_dist = distance_df.loc[distance_df['ending'] == cluster_name, :]
                    temp_dist = temp_dist.loc[temp_dist['starting'].isin(terminal_clusters) == False, :].copy()
                    temp_dist = temp_dist.sort_values("combined_score")
                    temp_dist.index = list(range(0, temp_dist.shape[0]))
                    my_G.add_edges_from([(temp_dist.loc[0, 'starting'], cluster_name)])
                except: 
                    print("Error: Failed to construct the cluster/cell state network. Make sure the initial cell state has the lowest average pseuodtime and terminal cell state has the highest average pseudotime. ")

        if cluster_name not in terminal_clusters: 
            out_nodes = [x[1] for x in my_G.out_edges(cluster_name)]
            if len(out_nodes) == 0: 
                try: 
                    temp_dist = distance_df.loc[distance_df['starting'] == cluster_name, :]
                    temp_dist = temp_dist.sort_values("combined_score")
                    temp_dist.index = list(range(0, temp_dist.shape[0]))
                    my_G.add_edges_from([(cluster_name, temp_dist.loc[0, 'ending'])])
                except: 
                    print("Error: Failed to construct the cluster/cell state network. Make sure the initial cell state has the lowest average pseuodtime and terminal cell state has the highest average pseudotime. ")
    return my_G 

def extract_trajectory(clusters_G, initial_clusters, terminal_clusters): 
    clusters_trajectory_dict = dict()
    i = 0 

    start_end_combos = itertools.product(initial_clusters, terminal_clusters)
    for unique_combo in start_end_combos:
        for item in nx.all_simple_paths(clusters_G, unique_combo[0], unique_combo[1]): 
            clusters_trajectory_dict["trajectory_" + str(i)] = item
            i = i + 1
    return clusters_trajectory_dict


