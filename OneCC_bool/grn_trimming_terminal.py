import networkx as net
import numpy as np 
import pandas as pd
from scipy.interpolate import UnivariateSpline

from .gene import *
from .OneCC_bool_simulator import *
from .network_structure import * 

def find_inconsistent_genes(initial_ss_dict, simulated_ss_dict, threshold_dict, target_ss_genes): 
    """Finding inconsistent genes between expected marker genes and marker genes from simulated expression profile 

    Args:
        initial_ss_dict (dictionary): dictionary representing the boolean profile for initial steady states. Keys are gene names and values are 0 or 1
        simulated_ss_dict (dictionary): dictionary representing the boolean profile for simulated steady states. Keys are gene names and values are 0 or 1
        threshold_dict (dictionary): dictionary representing the threshold. Keys are gene names and values are numericals
        target_ss_genes (list): list of genes that the simulation should express after perturbation 

    Returns:
        dictionary: dictionary representing genes that are still the same as initial steady state
    """
    diff_dict = dict()
    for temp_gene in simulated_ss_dict.keys():
        init_exp = int(initial_ss_dict[temp_gene] > threshold_dict[temp_gene])
        sim_exp = int(simulated_ss_dict[temp_gene] > threshold_dict[temp_gene])
        if init_exp == sim_exp: 
            if temp_gene not in target_ss_genes: # if the gene is not also expressed in target steady state
                diff_dict[temp_gene] = init_exp
    return diff_dict

# find priority gene based on reachability of the network 
def find_priority_gene(temp_grn_all, gene_list):

    diff_grn = temp_grn_all.loc[np.logical_and(temp_grn_all['TF'].isin(gene_list), temp_grn_all['TG'].isin(gene_list)), :]
    g = net.DiGraph()  

    for gene in gene_list: 
        g.add_node(gene)

    # create the networkx object 
    for temp_index in diff_grn.index:
        net.add_path(g, [diff_grn.loc[temp_index, "TF"],diff_grn.loc[temp_index, "TG"]])
    
    curr_reachability = 0
    priority_gene = ""
    for temp_gene in gene_list: 
        if len(net.shortest_path(g,temp_gene)) > curr_reachability: 
            priority_gene = temp_gene
            curr_reachability = len(net.shortest_path(g,temp_gene))
    return priority_gene

def bin_smooth(time_series, pseudoTime_bin, smooth_style = "median", spline_ME = 0.1):
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

    spline_w = np.divide(1, stand_dev)

    smoothed_data = pd.DataFrame()
    smoothed_data['PseudoTime'] = time_list
    smoothed_data['expression'] = smoothed_exp

    #spline_s = smoothed_data.shape[0] * (spline_ME ** 2)
    #spline_xy = UnivariateSpline(smoothed_data['PseudoTime'],smoothed_data['expression'], s = spline_s)
    #moothed_data['splined_exp'] = spline_xy(smoothed_data['PseudoTime'])
    return smoothed_data 

def find_earliest_activation(target_smoothed_time, thres):
    target_smoothed_time['activation'] = target_smoothed_time['expression'] > thres
    active_target_df = target_smoothed_time.loc[target_smoothed_time['activation'] == True, :]
    consec_num = 3
    
    if active_target_df.shape[0] == 0: 
        print("at no point is the gene expressed higher")
        return np.max(target_smoothed_time['PseudoTime'])
    elif active_target_df.shape[0] < consec_num: 
        print("less time points than 3")
        return np.max(active_target_df['PseudoTime'])
    else: 
        temp_time_index = active_target_df.index[0]
        for i in range(0, len(active_target_df.index) - (consec_num - 1)): 
            temp_time_index = active_target_df.index[i]
            if list(range(temp_time_index, temp_time_index + consec_num)) == list(active_target_df.index[i:i+3]):
                break
        return active_target_df.loc[temp_time_index, "PseudoTime"]

# currently it only considers the earliest activation 
# probably need to also consider the earliest suppression 

def trim_network_ss_anti(train_grn, ss_gene_sets, train_exp, train_st, trajectory_cells_dict, pseudoTime_bin, pt_col = 'pseudo_time', cluster_col = "cluster_label"):
    for temp_ss in ss_gene_sets.keys():
        temp_ss_genes = ss_gene_sets[temp_ss] # current genes 

        temp_grn_ss = train_grn.loc[np.logical_and(train_grn['TF'].isin(temp_ss_genes), train_grn['TG'].isin(temp_ss_genes)), :]

        # check if all the ss network is positively connected 
        # if they are not necessarily possitively connected, then make them positively connected 
        
        for edge_index in temp_grn_ss.index: 
            if train_grn.loc[edge_index, 'Type'] == "-":
                print("there might be a trouble in " + temp_grn_ss.loc[edge_index, 'TF'] + " ", temp_grn_ss.loc[edge_index, 'TG'])
                train_grn.loc[edge_index, 'Type'] == "+"
        
        # check if there are mutual antogonization 
        anti_ss_list = list(ss_gene_sets.keys())
        #anti_ss_list.remove(temp_ss) 

        for anti_ss in anti_ss_list:
            not_converge = True

            while not_converge == True:
                all_genes = list(ss_gene_sets[temp_ss]) + list(ss_gene_sets[anti_ss])
                temp_grn_all = train_grn.loc[np.logical_and(train_grn['TF'].isin(temp_ss_genes + ss_gene_sets[anti_ss]), train_grn['TG'].isin(temp_ss_genes + ss_gene_sets[anti_ss])), :]
                orphan_genes = np.setdiff1d(all_genes, np.array(temp_grn_all['TG']))

                orphan_grn = pd.DataFrame()
                orphan_grn['TF'] = orphan_genes
                orphan_grn['TG'] = orphan_genes
                orphan_grn['Type'] = "+"
                
                temp_grn_all = pd.concat([temp_grn_all, orphan_grn])

                MyNetwork = network_structure("network")
                MyNetwork.train_dummy_grn(temp_grn_all)
                
                # make the threshold dictionary to find between on and off 
                threshold_dict = dict()
                for temp_gene in MyNetwork.network_dict.keys():
                    threshold_dict[temp_gene] = (MyNetwork.network_dict[temp_gene].norm_factors['max'] - MyNetwork.network_dict[temp_gene].norm_factors['min']) / 2
                
                OneCC_sim = OneCC_bool_simulator()
                OneCC_sim.add_network_compilation(MyNetwork.subnet_name, MyNetwork)
                TFs = np.unique(temp_grn_all['TF'])
                OneCC_sim.TFs = TFs 

                #all_genes = np.unique(list(temp_grn_all['TF']) + list(temp_grn_all['TG']))
                init_dict = dict()
                perturb_dict = dict()

                for gene in all_genes: 
                    if gene in temp_ss_genes:
                        perturb_dict[gene] = 2
                    
                    if gene in ss_gene_sets[anti_ss]:
                        init_dict[gene] = 2
                    else:
                        init_dict[gene] = 0.02
                
                OneCC_sim.simulate_exp(init_dict, MyNetwork.subnet_name, perturb_dict, decay_rate = 0.1, num_sim = 90000, t_interval = 0.1, stochasticity = False)
                sim_exp = OneCC_sim.sim_exp
                sim_dict = sim_exp.iloc[:, -1].to_dict()
                inconsist_genes = find_inconsistent_genes(init_dict, sim_dict, threshold_dict, temp_ss_genes)

                if len(inconsist_genes) == 0: 
                    not_converge = False
                else: 
                    priority_gene = find_priority_gene(temp_grn_all, list(inconsist_genes.keys()))
                    # get lineage from the anti-steady state because the gene uis expressed in that lineage 
                    # find out when exact is that gene expressed in the lineage 
                    target_st = train_st.loc[train_st[cluster_col].isin(trajectory_cells_dict[anti_ss]), :]

                    target_time_series = pd.DataFrame()
                    target_time_series['expression'] = train_exp.loc[priority_gene, target_st.index]
                    target_time_series['PseudoTime'] = target_st[pt_col]
                    target_time_series.index = target_st.index

                    target_smoothed_time = bin_smooth(target_time_series, pseudoTime_bin, smooth_style = "median", spline_ME = 0.1)
                    target_act_time = find_earliest_activation(target_smoothed_time, threshold_dict[priority_gene])
                    
                    # potential activation 
                    priority_gene_df = train_grn.loc[train_grn['TG'] == priority_gene, :]
                   
                    poss_act_genes = ss_gene_sets[anti_ss]
                    poss_act_genes = [x for x in poss_act_genes if not x in list(priority_gene_df['TF'])]
                    poss_act_df = pd.DataFrame()
                    poss_act_time = list()
                    for temp_regulon in poss_act_genes:
                        regulon_st = train_st.loc[train_st[cluster_col].isin(trajectory_cells_dict[anti_ss]), :]

                        regulon_time_series = pd.DataFrame()
                        regulon_time_series['expression'] = train_exp.loc[temp_regulon, regulon_st.index]
                        regulon_time_series['PseudoTime'] = regulon_st[pt_col]
                        regulon_time_series.index = regulon_st.index

                        regulon_smoothed_time = bin_smooth(regulon_time_series, pseudoTime_bin, smooth_style = "median", spline_ME = 0.1)
                        temp_regulon_act_time = find_earliest_activation(regulon_smoothed_time, threshold_dict[temp_regulon])
                        
                        poss_act_time.append(temp_regulon_act_time)
                    poss_act_df['TF'] = poss_act_genes
                    poss_act_df['time'] = temp_regulon_act_time
                    poss_act_df['Type'] = "+"
                    
                    # potential suppression 
                    poss_sup_genes = ss_gene_sets[temp_ss]
                    poss_sup_genes = [x for x in poss_sup_genes if not x in list(priority_gene_df['TF'])]

                    poss_sup_df = pd.DataFrame()
                    poss_sup_time = list()
                    for temp_regulon in poss_sup_genes:
                        regulon_st = train_st.loc[train_st[cluster_col].isin(trajectory_cells_dict[temp_ss]), :]

                        regulon_time_series = pd.DataFrame()
                        regulon_time_series['expression'] = train_exp.loc[temp_regulon, regulon_st.index]
                        regulon_time_series['PseudoTime'] = regulon_st[pt_col]
                        regulon_time_series.index = regulon_st.index

                        regulon_smoothed_time = bin_smooth(regulon_time_series, pseudoTime_bin, smooth_style = "median", spline_ME = 0.1)
                        temp_regulon_sup_time = find_earliest_activation(regulon_smoothed_time, threshold_dict[temp_regulon])
                        
                        poss_sup_time.append(temp_regulon_sup_time)

                    poss_sup_df['TF'] = poss_sup_genes
                    poss_sup_df['time'] = poss_sup_time
                    poss_sup_df['Type'] = "-"

                    poss_df = pd.concat([poss_act_df, poss_sup_df])
                    poss_df['time_diff'] = np.abs(poss_df['time'] - target_act_time)

                    poss_df = poss_df.sort_values("time_diff")

                    # if there are no suitable edge to be added 
                    if poss_df.shape[0] == 0:
                        not_converge = False
                        break 

                    additional_df = pd.DataFrame(data = [[poss_df.iloc[0, 0], priority_gene, poss_df.iloc[0, 2]]], columns = ['TF', 'TG', 'Type'])
                    additional_df.index = [str(poss_df.iloc[0, 0] + "_" + priority_gene + "_new")]
                    train_grn = pd.concat([train_grn, additional_df])
    return train_grn


def add_self_ss_edge(train_grn, ss_gene_sets):
    for temp_ss in ss_gene_sets.keys():
        temp_ss_genes = ss_gene_sets[temp_ss]
        
        # for isolated gene, add in self-regulation 

        not_converge = True

        while not_converge == True: 
            MyNetwork = network_structure("network")
            temp_grn_ss = train_grn.loc[np.logical_or(train_grn['TF'].isin(temp_ss_genes), train_grn['TG'].isin(temp_ss_genes)), :]

            MyNetwork.train_dummy_grn(temp_grn_ss)
            
            threshold_dict = dict()
            for temp_gene in MyNetwork.network_dict.keys():
                threshold_dict[temp_gene] = (MyNetwork.network_dict[temp_gene].norm_factors['max'] - MyNetwork.network_dict[temp_gene].norm_factors['min']) / 2

            OneCC_sim = OneCC_bool_simulator()
            OneCC_sim.add_network_compilation(MyNetwork.subnet_name, MyNetwork)
            TFs = np.unique(temp_grn_ss['TF'])
            OneCC_sim.TFs = TFs 

            init_dict = dict()
            perturb_dict = dict()

            all_genes = np.unique(list(temp_grn_ss['TF']) + list(temp_grn_ss['TG']))
            for gene in all_genes:
                if gene in temp_ss_genes: 
                    init_dict[gene] = 2
                else:
                    init_dict[gene] = 0
                    perturb_dict[gene] = -2
            
            OneCC_sim.simulate_exp(init_dict, MyNetwork.subnet_name, perturb_dict, decay_rate = 0.1, num_sim = 90000, t_interval = 0.1, stochasticity = False)
            sim_exp = OneCC_sim.sim_exp
            sim_dict = sim_exp.iloc[:, -1].to_dict()

            # find genes that should be self regulated 
            genes_self_reg = list()
            for gene in sim_dict.keys():
                if gene in temp_ss_genes:
                    if sim_dict[gene] < threshold_dict[gene]:
                        
                        # check if the gene is already auto regulated 
                        new_temp_grn_ss = temp_grn_ss.loc[temp_grn_ss['TF'] == gene, :]
                        new_temp_grn_ss = new_temp_grn_ss.loc[temp_grn_ss['TG'] == gene, :]
                        if new_temp_grn_ss.shape[0] == 0:
                            genes_self_reg.append(gene)
            
            if len(genes_self_reg) == 0:
                not_converge = False
            else:
                priority_gene = find_priority_gene(temp_grn_ss, genes_self_reg)
                additional_df = pd.DataFrame(data = [[priority_gene, priority_gene, "+"]], columns = ['TF', 'TG', 'Type'])
                additional_df.index = [str(priority_gene + "_" + priority_gene + "_new")]
                train_grn = pd.concat([train_grn, additional_df])
    return train_grn 

def define_ss_genes(train_exp, train_st, trajectory_cells_dict, bool_thresholds, pt_col = 'pseudoTime', cluster_col = 'cluster_label', n_cells = 50): 
    ss_genes_dict = dict()
    for lineage in trajectory_cells_dict.keys():
        sub_clusters = trajectory_cells_dict[lineage] 
        sub_train_st = train_st.loc[train_st[cluster_col].isin(sub_clusters), :]
        sub_train_exp = train_exp.loc[:, sub_train_st.index]

        sub_train_st = sub_train_st.sort_values(by = pt_col, ascending = False)
        sub_train_st = sub_train_st.iloc[0:n_cells, :]
        sub_train_exp = sub_train_exp.loc[:, sub_train_st.index]

        turned_on_genes = sub_train_exp.mean(axis = 1) > bool_thresholds
        ss_genes_dict[lineage] = list(turned_on_genes[turned_on_genes == True].index)

    return ss_genes_dict