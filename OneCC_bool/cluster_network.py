import networkx as nx 
import numpy as np 
import pandas as pd 

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
            distance_df = distance_df.append(temp_dict, ignore_index = True)

    my_G = nx.DiGraph()
    for cluster_name in mean_exp.columns:
        my_G.add_node(cluster_name)
        
        # if the cluster name is the inital 
        if cluster_name not in initial_clusters:
            # no need to check incoming 
            incoming_nodes = [x[0] for x in my_G.in_edges(cluster_name)]
            if len(incoming_nodes) == 0: 
                temp_dist = distance_df.loc[distance_df['ending'] == cluster_name, :]
                temp_dist = temp_dist.sort_values("distance")
                my_G.add_edges_from([(temp_dist.iloc[0, 0], cluster_name)])
                
        if cluster_name not in terminal_clusters: 
            out_nodes = [x[1] for x in my_G.out_edges(cluster_name)]
            if len(out_nodes) == 0: 
                temp_dist = distance_df.loc[distance_df['starting'] == cluster_name, :]
                temp_dist = temp_dist.sort_values("distance")
                my_G.add_edges_from([(cluster_name, temp_dist.iloc[0, 1])])

    return my_G 


def extract_lineage(clusters_G, initial_clusters, terminal_clusters): 
    clusters_lineage_dict = dict()
    i = 0 
    for item in nx.all_simple_paths(clusters_G, initial_clusters, terminal_clusters): 
        clusters_lineage_dict["lineage_" + str(i)] = item
        i = i + 1
    
    return clusters_lineage_dict

def extract_mutual_inhibiton(clusters_G, initial_clusters, terminal_clusters):
    conflicting_clusters = dict()

    # find the cluster that needs mutual inhibition 
    for temp_node in clusters_G.nodes():
        if len(list(clusters_G.out_edges(temp_node))) >= 2: 
            for temp_edge in list(clusters_G.out_edges(temp_node)):
                conflicting_clusters[temp_edge[1]] = list()
    
    # all the clusters in that lineage 
    for conflict_cluster in conflicting_clusters.keys():
        for temp_lineage in list(nx.all_simple_paths(clusters_G, initial_clusters, terminal_clusters)): 
            if conflict_cluster in temp_lineage: 
                conflicting_clusters[conflict_cluster] = list(np.unique(conflicting_clusters[conflict_cluster] + temp_lineage))
    
    return conflicting_clusters


