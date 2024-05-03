import numpy as np 
import pandas as pd
import itertools
import networkx as nx
import anndata
import scipy as sp
from .genetic_algorithm_GRN_trimming import define_states
from .genetic_algorithm_GRN_trimming import define_transition
from .genetic_algorithm_GRN_trimming import curate_training_data
from .genetic_algorithm_GRN_trimming import calc_corr # can we replace with x.T.corr()?
from .genetic_algorithm_GRN_trimming import create_network_ensemble
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def find_gene_change_trajectory(train_exp, train_st, trajectory_cells_dict, cluster_col, pt_col, bool_thresholds, pseudoTime_bin, smooth_style = 'mean'):
    """Identify the approximate pseudotime at which the genes change status (on to off or off to on) along each trajectory. 
    This information will be used to help curate the training data for genetic algorithm onesc.curate_training_data(). 


    Args:
        train_exp (pandas.DataFrame): Single-cell expression table. 
        train_st (pandas.DataFrame): Sample table for the expression table. 
        trajectory_cells_dict (dict): Output from onesc.extract_trajectory(). It is a dictionary that categorizes different cell states into different trajectory. 
                                      The name of the dictionary is the trajectory name and the item of the dictionary is a list of cell state. 
        cluster_col (str): The column name of the column in train_st containing cell states or clusters information. 
        pt_col (str): The column name of the column in train_st containing psedotime ordering. 
        bool_thresholds (pandas.series): Output from onesc.find_threshold_vector. It contains the threshold value (between on and off) for each gene. The indexes are the gene names and the values are the threshold expression values. 
        pseudoTime_bin (float): The pseudotime resolution at which we bin the cells within and average the expressions. 
        smooth_style (str, optional): The smoothing style ('mean' or 'median'). Defaults to 'mean'.

    Returns:
        dict: A dictionary of pandas dataframes documenting the pseudotime point at which the genes change status across a trajectory. The keys of the dictionary represent the trajectory names, and the items are pandas dataframe document the pseudotime point at which genes change status. 
    """
    trajectory_time_change_dict = dict()
    for temp_trajectory in trajectory_cells_dict.keys(): 
        activation_time = find_genes_time_change(train_st, train_exp, trajectory_cells_dict, cluster_col, pt_col, bool_thresholds, pseudoTime_bin, temp_trajectory, smooth_style = smooth_style)
        trajectory_time_change_dict[temp_trajectory] = activation_time
    return trajectory_time_change_dict

def find_genes_time_change(train_st, train_exp, trajectory_cluster_dict, cluster_col, pt_col, vector_thresh, pseudoTime_bin, trajectory_name, smooth_style = 'mean'):
    """Find the pseudotime point at which the genes change status (on to off or off to on) along one trajectory. This is a helper function for onesc.find_gene_change_trajectory

    Args:
        train_st (pandas.DataFrame): Sample table for the expression table. 
        train_exp (pandas.DataFrame): Single-cell expression table. 
        trajectory_cells_dict (dict): Output from onesc.extract_trajectory(). It is a dictionary that categorizes different cell states into different trajectory. 
                                      The name of the dictionary is the trajectory name and the item of the dictionary is a list of cell state. 
        cluster_col (str): The column name of the column in train_st containing cell states or clusters information. 
        pt_col (str): The column name of the column in train_st containing psedotime ordering. 
        vector_thresh (pandas.series): Output from onesc.find_threshold_vector. It contains the threshold value (between on and off) for each gene. The indexes are the gene names and the values are the threshold expression values. 
        pseudoTime_bin (float): The pseudotime resolution at which we bin the cells within and average the expressions. 
        trajectory_name (str): The name of the trajectory. 
        smooth_style (str, optional): The smoothing style ('mean' or 'median'). Defaults to 'mean'.

    Returns:
        pandas.DataFrame: pandas dataframe document the pseudotime point at which genes change status.
    """
    target_clusters = trajectory_cluster_dict[trajectory_name]
    sub_train_st = train_st.loc[train_st[cluster_col].isin(target_clusters), :]
    sub_train_st = sub_train_st.sort_values(pt_col)
    sub_train_exp = train_exp.loc[:, sub_train_st.index]
    activation_time_df = pd.DataFrame()
    for temp_gene in sub_train_exp.index: 
        time_series = pd.DataFrame()
        time_series['expression'] = sub_train_exp.loc[temp_gene, :]
        time_series['PseudoTime'] = sub_train_st[pt_col]
        smoothed_series = bin_smooth(time_series, pseudoTime_bin, smooth_style = smooth_style)
        smoothed_series['boolean'] = smoothed_series['expression'] > vector_thresh[temp_gene]
        temp_change_df = find_transition_time(smoothed_series, temp_gene)
        activation_time_df = pd.concat([activation_time_df, temp_change_df])
    activation_time_df = activation_time_df.sort_values("PseudoTime")
    activation_time_df.index = list(range(0, activation_time_df.shape[0]))
    return activation_time_df

def find_transition_time(smoothed_series, temp_gene):
    """Find the transition pseudotime point. This is a helper function for onesc.find_genes_time_change()

    Args:
        smoothed_series (pandas.DataFrame): pandas dataframe containing smoothed expression values for a particular gene at different pseudotime interval. 
        temp_gene (str): the name of the gene 

    Returns:
        pandas.DataFrame: pandas dataframe documenting all the instance at which the temp_gene changed across pseudotime. 
    """
    current_status = smoothed_series.loc[0, 'boolean'] # identify the inital state of the gene 
    transition_df = pd.DataFrame(index = ['PseudoTime', 'gene', 'type'])
    lag = 5
    if current_status == True: 
        transition_df = pd.concat([transition_df, pd.Series({'PseudoTime': 0, 'gene': temp_gene, 'type': '+'})], axis = 1)
    elif current_status == False: 
        transition_df = pd.concat([transition_df, pd.Series({'PseudoTime': 0, 'gene': temp_gene, 'type': '-'})], axis = 1)
    for index in smoothed_series.index[0:len(smoothed_series.index)-lag]: 
        if smoothed_series.loc[index, 'boolean'] == current_status:
            continue
        else: 
            if np.sum(smoothed_series.loc[index:index+lag, 'boolean']) == 0 or np.sum(smoothed_series.loc[index:index+lag, 'boolean']) == len(smoothed_series.loc[index:index+lag, 'boolean']):
                # identify whether if there is a positive change or negative change
                if current_status == False and smoothed_series.loc[index, 'boolean'] == True: 
                    reg_type = "+"
                elif current_status == True and smoothed_series.loc[index, 'boolean'] == False: 
                    reg_type = "-"
                transition_df = pd.concat([transition_df, pd.Series({'PseudoTime': smoothed_series.loc[index, 'PseudoTime'], 'gene': temp_gene, 'type': reg_type})], axis = 1)
                current_status = smoothed_series.loc[index, 'boolean']
    transition_df = transition_df.T
    transition_df.index = list(range(0, transition_df.shape[0]))
    return transition_df

def bin_smooth(time_series, pseudoTime_bin, smooth_style = "mean"):
    """Bin average the expression across pseudotime bins. This is a helper function for onesc.find_genes_time_change

    Args:
        time_series (pandas.DataFrame): Dataframe with one column representing pseudotime and other column representing the expression of a cell. 
        pseudoTime_bin (float): Bin size for pseudotime. 
        smooth_style (str, optional): Whether to get the 'mean' of the expression within pseudotime bin or 'median'. Defaults to "mean".

    Returns:
        pandas.DataFrame: Dataframe with bin averged or medianed expressions across different pseudotime segments. 
    """
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

    smoothed_data = pd.DataFrame()
    smoothed_data['PseudoTime'] = time_list
    smoothed_data['expression'] = smoothed_exp
    return smoothed_data 

def find_threshold_vector(exp_df, samp_st, cluster_col = "cluster", cutoff_percentage = 0.4): 
    """Identify the expression threshold for each gene. Compile them in a series. 
       First, it calculates the average expression for each cell state or cluster. 
       Then, it identifies the highest averge expression and lowest expression. 
       cutoff_percentage * the difference between highest and lowest average expression would be the expression threshold for the gene. 

    Args:
        exp_df (pandas.DataFrame): The expression matrix. 
        samp_st (pandas.DataFrame): The sample table for the single-cell expression matrix. 
        cluster_col (str, optional): The column name for the column containing the cluster information. Defaults to "cluster".
        cutoff_percentage (float, optional): The minimum percent cut-off of the difference between highest and lowest average experssion. Defaults to 0.4.

    Returns:
        pandas.Series: Series of expression threshold for the genes. 
    """
    cluster_exp = pd.DataFrame()
    for temp_cluster in np.unique(samp_st[cluster_col]):
        temp_st = samp_st.loc[samp_st[cluster_col] == temp_cluster, :]
        temp_exp = exp_df.loc[:, temp_st.index]
        cluster_exp[temp_cluster] = temp_exp.mean(axis = 1)
    return ((cluster_exp.max(axis = 1) - cluster_exp.min(axis = 1)) * cutoff_percentage + cluster_exp.min(axis = 1))

def construct_cluster_network(train_exp, sampTab, initial_clusters, terminal_clusters, cluster_col = "cluster_id", pseudo_col = "pseudotime"):
    """Construct a graph connecting the cell states or clusters transitions based on expression similarities and pseudotime ordering (lower pseudotime suggests higher upstream). 
       This function typically works the best when the transitional relationships are not cyclical (ex A->B->C->A). 
       If in any event where this function cannot create the cluster network, the user can also manually create a networkx directed graph object. 

    Args:
        train_exp (pandas.DataFrame): The single-cell expression table. 
        sampTab (pandas.DataFrame): The sample table for the single-cell expression table. 
        initial_clusters (list): A list of user defined initial cell states or clusters. 
        terminal_clusters (list): A list of user defined terminal cell states or clusters. 
        cluster_col (str, optional): The column name for column in the sample table containing cell states information. Defaults to "cluster_id".
        pseudo_col (str, optional): The column name for column containing pseudotime information. Defaults to "pseudotime".

    Returns:
        networkx.DiGraph: A networkx directed graph object summarizing the transitional relationship between cell states. 
    """
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

    # calculate the euclidean distance between clusters 
    distance_df = pd.DataFrame() 
    for i in range(mean_exp.shape[1]):
        for j in range(i + 1, mean_exp.shape[1]):
            temp_dict = {"starting": mean_exp.columns[i], "ending": mean_exp.columns[j], "distance": np.linalg.norm(mean_exp.iloc[:, j] - mean_exp.iloc[:, i])}
            distance_df = pd.concat([distance_df, pd.DataFrame([temp_dict])], ignore_index = True)
    distance_df['norm_dist'] = (distance_df['distance'] - np.min(distance_df['distance'])) / (np.max(distance_df['distance']) - np.min(distance_df['distance']))
    distance_df['starting_pt'] = None
    mean_pt.index = mean_pt['cluster']

    # calculate the combined score 
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
    """Deconstruct the networkx dircted graph object into a individual trajectories. The key of the dictionary is the trajectory name, and the items are lists of clusters inside that trajectory. 

    Args:
        clusters_G (networkx.DiGraph): Networkx directed graph object from onesc.construct_cluster_network or manually curated. 
        initial_clusters (list): A list of user defined initial cell states or clusters. 
        terminal_clusters (list): A list of user defined terminal cell states or clusters. 

    Returns:
        dict: Dictionary of different trajectories and the clusters associated to the individual trajectory. 
    """
    clusters_trajectory_dict = dict()
    i = 0 
    start_end_combos = itertools.product(initial_clusters, terminal_clusters)
    for unique_combo in start_end_combos:
        for item in nx.all_simple_paths(clusters_G, unique_combo[0], unique_combo[1]): 
            clusters_trajectory_dict["trajectory_" + str(i)] = item
            i = i + 1
    return clusters_trajectory_dict



