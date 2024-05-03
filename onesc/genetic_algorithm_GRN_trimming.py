import numpy as np 
import pandas as pd
import pygad
import itertools
from joblib import Parallel, delayed, cpu_count
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

def define_states(exp_tab, samp_tab, trajectory_cluster, vector_thresh, cluster_col = 'cluster_id', percent_exp = 0.3):
    """Define the cell state boolean profiles for each of the trajectory. 

    Args:
        exp_tab (pandas.DataFrame): The single-cell expression profiles. 
        samp_tab (pandas.DataFrame): The sample table for the single-cell expression profiles. 
        trajectory_cluster (dict): The output from onesc.extract_trajectory. It is a dictionary detailing the clusters along a trajectory for all trajectories. 
        vector_thresh (pandas.Series): The output from onesc.find_threshold_vector. It is a pandas Series of the expression threshold for all the genes. 
        cluster_col (str, optional): The column name of the column in the sample table with cell state/cluster information. Defaults to 'cluster_id'.
        percent_exp (float, optional): The minimum percent expression of cells in the cell state/cluster for a gene to be considered as ON even if the average expression passes the expression threshold. Defaults to 0.2.
    
    Returns:
        dict: A dictionary containing all the Boolean gene states for different cell state/clusters in a trajectory for all trajectories. 
    """
    def check_zero(vector):
        return (len(vector) - np.sum(vector == 0)) / len(vector)
    state_dict = dict()
    for trajectory in trajectory_cluster.keys():
        temp_df = pd.DataFrame()
        for cell_type in trajectory_cluster[trajectory]:
            sub_st = samp_tab.loc[samp_tab[cluster_col] == cell_type, :]
            sub_exp = exp_tab.loc[:, sub_st.index]
            temp_df[cell_type] = np.logical_and((sub_exp.mean(axis = 1) >= vector_thresh), (sub_exp.apply(check_zero, axis = 1) >= percent_exp)) * 1
        state_dict[trajectory] = temp_df
    return state_dict  

def define_transition(state_dict):
    """Documenting the cell states at which genes transition (ON to OFf or OFF to ON). 

    Args:
        state_dict (dict): The output from onesc.define_states. It is a dictionary containig all the Boolean gene states for different cell state/clusters in a trajectory for all trajectories. 

    Returns:
        dict: A dictionary containing all the cell states at which genes transition from ON to OFF or OFF to ON. 
    """
    transition_dict = dict()
    for trajectory in state_dict.keys():
        temp_df = pd.DataFrame()
        initial_condition = True
        prev_col = None 
        for cell_type in state_dict[trajectory].columns:
            if initial_condition == True: 
                temp_trans = state_dict[trajectory][cell_type]
                temp_df[cell_type] = temp_trans.replace(0, -1)
                prev_col = cell_type
                initial_condition = False
            else: 
                temp_trans = state_dict[trajectory][cell_type] - state_dict[trajectory][prev_col]
                temp_df[cell_type] = temp_trans
                prev_col = cell_type
                
        transition_dict[trajectory] = temp_df
    return transition_dict

def curate_training_data(state_dict, transition_dict, trajectory_time_change_dict, samp_tab, cluster_id = "cluster_id", pt_id = "dpt_pseudotime",act_tolerance = 0.01, selected_regulators = list()):
    """Compile training data before running with genetic algorithm. This curates the expression states of the gene (as label) and the expression states (as features) of all transcription regulators at different cell states. 
       In the event when the gene and a transcription regulator both change at the same cell staste, pseudotime ordering at which the gene and transcription regulator change will be considered when curating gene expression states (label) and transcription regulators expression states (features). 
       For instance, if transcription regulator change way ahead of target gene change (< target gene change - act_tolerance), then for curating the feature matrix, we would use the post changed status of the transcription regulator. 
       We will also output unlikely activator for each gene based on finding transcription factors that are inactive at the time when the gene is first activated. We also included an empty array for unlikely repressors in case users want to add them in manually. 
       
    Args:
        state_dict (dict): The output from onesc.define_states. It is a dictionary containing all the Boolean gene states for different cell state/clusters in a trajectory for all trajectories. 
        transition_dict (dict): The output from onesc.define_transition. It is a dictionary containing all the cell states at which genes transition from ON to OFF or OFF to ON. 
        trajectory_time_change_dict (dict): The output from onesc.find_gene_change_trajectory. It is a dictionary of pandas dataframes documenting the pseudotime point at which the genes change status across a trajectory. The keys of the dictionary represent the trajectory names, and the items are pandas dataframe document the pseudotime point at which genes change status. 
        samp_tab (pandas.DataFrame): he sample table for the single-cell expression profiles. 
        cluster_id (str, optional): The column name of the column in the sample table with cell state/cluster information. Defaults to 'cluster_id'.
        pt_id (str, optional): The column name of the column in the sample table with pseudotime information. Defaults to "dpt_pseudotime".
        act_tolerance (float, optional): The range of activation window (pseudotime units). If both target gene and regulator change in the same cell state and if the pseudotime at which the regulator change is < target gene change - act_tolerance, then in the curated gene state in the feature matrix of the regulator would use the postchanged regulator state. Defaults to 0.01.
        selected_regulators (list, optional): The list of regulators or transcription factors. Defaults to list(). If input an empty list, then assume all genes are capable of transcriptional regulations. 

    Returns:
        dict: A dictionary of feature matrix, gene status label, unlikely activators and unlikely repressors for each gene ready for genetic algorithm optimization. 
    """
    all_genes = transition_dict['trajectory_0'].index

    def extract_steady_states(state_df, target_gene, trajectory_name):
        extract_ss_df = state_df.iloc[:, -1:].copy()
        extract_ss_df.columns = extract_ss_df.columns + "_SS_" + trajectory_name
        target_gene_state = [extract_ss_df.loc[target_gene, extract_ss_df.columns[0]]]
        prior_df = state_df.iloc[:, -2]
        if prior_df[target_gene] == 1:
            extract_ss_df.loc[target_gene, extract_ss_df.columns[0]] = 1
        else:
            extract_ss_df.loc[target_gene, extract_ss_df.columns[0]] = 0
        return [target_gene_state, extract_ss_df]

    def extract_stable_state(state_df, target_gene, unstable_states_list, trajectory_name):
        if len(unstable_states_list) == 0:
            #extract_stable_df = state_df.drop(state_df.columns[np.array([0, state_df.shape[1] - 1])], axis = 1).copy()
            extract_stable_df = state_df.drop(state_df.columns[int(state_df.shape[1] - 1)], axis = 1).copy()
        else:
            #exclude_index_list = np.array([0])
            exclude_index_list = np.array([])
            exclude_index_list = np.append(exclude_index_list, int(state_df.shape[1] - 1))
            exclude_index_list = np.unique(exclude_index_list)
            if len(exclude_index_list) > 1:
                extract_stable_df = state_df.drop(state_df.columns[exclude_index_list], axis = 1).copy()
            elif len(exclude_index_list) == 1:
                extract_stable_df = state_df.drop(state_df.columns[int(exclude_index_list[0])], axis = 1).copy()
            extract_stable_df = extract_stable_df.loc[:, extract_stable_df.columns.isin(unstable_states_list) == False].copy()

        # if there are no stable state 
        if extract_stable_df.shape[1] == 0:
            return [[], pd.DataFrame()]
        
        target_gene_state = list(extract_stable_df.loc[target_gene, :])
        for state in extract_stable_df.columns:
            prev_index = np.where(state_df.columns == state)[0][0] - 1
            if prev_index < 0:
                continue
            else:
                prior_state = state_df.iloc[:, prev_index]
                if prior_state[target_gene] == 1:
                    extract_stable_df.loc[target_gene, state] = 1
                else:
                    extract_stable_df.loc[target_gene, state] = 0
        extract_stable_df.columns = extract_stable_df.columns + "_stable_" + trajectory_name
        return [target_gene_state, extract_stable_df]

    def check_stable_initial(trans_dict, target_gene):
        for temp_trajectory in trans_dict.keys():
            temp_transition = trans_dict[temp_trajectory]
            if 0 in np.where(temp_transition.loc[target_gene, :] != 0)[0] - 1:
                return False
        return True

    def extract_stable_initial_state(state_dict, target_gene, initial_stability = True):
        if initial_stability == True:
            temp_state = state_dict['trajectory_0']
            extract_ss_initial_df = temp_state.iloc[:, 0:1].copy()
            extract_ss_initial_df.columns = extract_ss_initial_df.columns + "_initial_SS_all"
            target_gene_state = [extract_ss_initial_df.loc[target_gene, extract_ss_initial_df.columns[0]]]
            return [target_gene_state, extract_ss_initial_df]
        else:
            return [[], pd.DataFrame()]

    def extract_unstable_state(state_df, transition_df, time_change_df, target_gene, exclude_index_list, samp_tab, cluster_id, pt_id, trajectory_name, act_tolerance = 0.01):
        target_gene_state = []
        extract_unstable_df = pd.DataFrame()
        more_unlikely_activators = []

        def find_potential_regulators(transition_df, time_change_df, target_gene, exclude_index, cluster_id, pt_id, act_tolerance):
            changing_genes = transition_df.index[transition_df.iloc[:, exclude_index + 1] != 0]
            potential_regulators = []
            min_time = np.median(samp_tab.loc[samp_tab[cluster_id] == transition_df.columns[exclude_index], pt_id]) 
            max_time = np.max(samp_tab.loc[samp_tab[cluster_id] == transition_df.columns[exclude_index + 1], pt_id]) 
            temp_time_change_df = time_change_df.loc[np.logical_and(time_change_df['PseudoTime'] >= min_time, time_change_df['PseudoTime'] <= max_time), :]
            target_gene_status = transition_df.iloc[:, exclude_index + 1][target_gene]
            
            if target_gene_status == 1:
                target_gene_act_df = temp_time_change_df.loc[np.logical_and(temp_time_change_df['gene'] == target_gene, temp_time_change_df['type'] == "+"), :]
            else:
                target_gene_act_df = temp_time_change_df.loc[np.logical_and(temp_time_change_df['gene'] == target_gene, temp_time_change_df['type'] == "-"), :]
            if target_gene_act_df.shape[0] == 0:
                target_gene_act = min_time # the change occured earlier than median of previous cluster
            else:
                target_gene_act = target_gene_act_df.iloc[target_gene_act_df.shape[0] - 1, :]['PseudoTime']
            
            for temp_changing_gene in changing_genes:
                regulator_gene_status = transition_df.iloc[:, exclude_index + 1][temp_changing_gene]
                if regulator_gene_status == 1:
                    regulator_gene_act_df = temp_time_change_df.loc[np.logical_and(temp_time_change_df['gene'] == temp_changing_gene, temp_time_change_df['type'] == "+"), :]
                else:
                    regulator_gene_act_df = temp_time_change_df.loc[np.logical_and(temp_time_change_df['gene'] == temp_changing_gene, temp_time_change_df['type'] == "-"), :]
                if regulator_gene_act_df.shape[0] == 0:
                    regulator_gene_act = min_time # the change occured earlier than median of previous cluster
                else:
                    regulator_gene_act = regulator_gene_act_df.iloc[regulator_gene_act_df.shape[0] - 1, :]['PseudoTime']
                if regulator_gene_act < target_gene_act - act_tolerance:
                    potential_regulators.append(temp_changing_gene)
            return potential_regulators
        
        if len(exclude_index_list) == 0:
            return [[], pd.DataFrame(), [], []]
        for temp_index in exclude_index_list: 
            target_gene_state.append(state_df.iloc[:, temp_index + 1][target_gene])
            temp_extract_unstable_df = state_df.iloc[:, temp_index:temp_index + 1].copy()
            potential_regulators = find_potential_regulators(transition_df, time_change_df, target_gene, temp_index, cluster_id, pt_id, act_tolerance)
            for potential_regulator in potential_regulators:
                temp_extract_unstable_df.loc[potential_regulator, temp_extract_unstable_df.columns[0]] = state_df.loc[potential_regulator, state_df.columns[temp_index + 1]]
            
            # get the unlikely activators 
            if state_df.iloc[:, temp_index + 1][target_gene] == 1: 
                for temp_regulator in state_df[state_df.columns[temp_index + 1]].index:
                    if temp_regulator in potential_regulators:
                        if state_df.loc[temp_regulator, state_df.columns[temp_index + 1]] == 1:
                            more_unlikely_activators.append(temp_regulator)

            extract_unstable_df = pd.concat([extract_unstable_df, temp_extract_unstable_df], axis = 1)
        
        # mark down the states that is unstable in any of the trajectory 
        if len(extract_unstable_df.columns) > 0: 
            unstable_states = list(extract_unstable_df.columns)
        else: 
            unstable_states = []
        extract_unstable_df.columns = extract_unstable_df.columns + "_unstable_" + trajectory_name 
        return [target_gene_state, extract_unstable_df, unstable_states, more_unlikely_activators]

    def find_unlikely_activators(state_dict, gene_interest):
        good_genes = np.array([])
        init_status = True
        for temp_trajectory in state_dict.keys():
            temp_state = state_dict[temp_trajectory]
            all_genes = np.array(temp_state.index)
            pos_columns_index = np.where(temp_state.loc[gene_interest, :] == 1)[0]
            if len(pos_columns_index) == 0:
                continue
            else: 
                temp_index = pos_columns_index[0]
            
            temp_good_genes = np.array(list(temp_state.index[temp_state.iloc[:, temp_index] == 1]))
           
            if temp_index == 0: # if the first occruance is at the inital, then it should be activated by itself or something outside of the network
                return all_genes

            if init_status == True:
                good_genes = np.array(temp_good_genes)
                init_status = False
            else:
                good_genes = np.intersect1d(good_genes, temp_good_genes)

        bad_genes = np.setdiff1d(all_genes, good_genes)
        return bad_genes

    def check_feature_mat(feature_mat, col_index, gene_status):
        def to_string_pattern(gene_pattern):
            gene_pattern = [str(x) for x in gene_pattern]
            return "_".join(gene_pattern)
        remove_index = []
        # categorize all the same gene pattern profiles into one 
        same_pattern_dict = dict()
        for temp_index in col_index:
            string_pattern = to_string_pattern(feature_mat.iloc[:, temp_index])
            if string_pattern not in same_pattern_dict.keys():
                same_pattern_dict[string_pattern] = [temp_index]
            else:
                same_pattern_dict[string_pattern] = same_pattern_dict[string_pattern] + [temp_index]
        for temp_key in same_pattern_dict.keys():
            temp_index_list = same_pattern_dict[temp_key]
            sub_gene_status = np.array(gene_status)[temp_index_list]
            if len(np.unique(sub_gene_status)) > 1:
                remove_index = remove_index + temp_index_list
        return remove_index

    def remove_conflicts(feature_mat, gene_status):
        filter_df  = pd.DataFrame(data = None, index = feature_mat.columns, columns = ['ct', 'style', 'trajectory', 'type'])   
        for temp_state in filter_df.index:
            filter_df.loc[temp_state, :] = temp_state.split("_")
        filter_df['col_index'] = list(range(0, feature_mat.shape[1]))
        state_meta = filter_df.copy()
        filter_df = filter_df.loc[filter_df['style'] != 'SS', :].copy()
        dup_states = filter_df['ct'].value_counts().index[filter_df['ct'].value_counts() > 1]
        
        remove_index_list = []
        for temp_state in dup_states: 
            sub_filter_df = filter_df.loc[filter_df['ct'] == temp_state, :]
            temp_remove_index_list = check_feature_mat(feature_mat, list(sub_filter_df['col_index']), gene_status)
            remove_index_list = remove_index_list + temp_remove_index_list
        
        
        if len(remove_index_list) > 1: 
            print(temp_gene + " have conflicting states, the below states are deleted")
            print(list(feature_mat.columns[remove_index_list]))
        
        new_feature_mat = feature_mat.drop(feature_mat.columns[remove_index_list], axis = 1).copy()
        good_index = list(state_meta.loc[new_feature_mat.columns, :]['col_index'])
        new_gene_status = np.array(gene_status)[good_index].copy()
        new_gene_status = list(new_gene_status)
        return [new_feature_mat, new_gene_status]
    
    training_dict = dict()
    for temp_gene in all_genes: 

        # find the unlike activators and repressors 
        unlikely_activators = find_unlikely_activators(state_dict, temp_gene)
        unlikely_repressors = np.array([])

        # The below will be for curate the training data 
        gene_status_label = []
        feature_mat = pd.DataFrame()
        gene_train_dict = dict()

        for temp_trajectory in transition_dict.keys():
            temp_transition = transition_dict[temp_trajectory] # transition matrix 
            temp_state = state_dict[temp_trajectory] # state matrix 
            temp_time_change = trajectory_time_change_dict[temp_trajectory] # time series 
            
            # find the index of the state before transition of temp_gene 
            # the states that were right before a transition is not stable 
            prev_index_list = np.where(temp_transition.loc[temp_gene, :] != 0)[0] - 1
            prev_index_list = prev_index_list[prev_index_list > -1]
            prev_index_list = [int(x) for x in prev_index_list]
            # get the unstable states
            [temp_label, temp_feature, unstable_states, more_unlikely_activators] = extract_unstable_state(temp_state, temp_transition, temp_time_change, temp_gene, prev_index_list, samp_tab, cluster_id, pt_id, temp_trajectory, act_tolerance)
            gene_status_label = gene_status_label + temp_label
            feature_mat = pd.concat([feature_mat, temp_feature], axis = 1).copy()
            unlikely_activators = np.concatenate((unlikely_activators, more_unlikely_activators))
            
            # get stable states. aka genes in states that do not change immediately 
            [temp_label, temp_feature] = extract_stable_state(temp_state, temp_gene, unstable_states, temp_trajectory)
            gene_status_label = gene_status_label + temp_label
            feature_mat = pd.concat([feature_mat, temp_feature], axis = 1).copy()
        
            # get the steady states 
            [temp_label, temp_feature] = extract_steady_states(temp_state, temp_gene, temp_trajectory)
            gene_status_label = gene_status_label + temp_label
            feature_mat = pd.concat([feature_mat, temp_feature], axis = 1).copy()

        if len(selected_regulators) != 0:
            feature_mat = feature_mat.loc[selected_regulators, :]

        [new_feature_mat, new_gene_status] = remove_conflicts(feature_mat, gene_status_label)
        gene_train_dict['feature_matrix'] = new_feature_mat.copy()
        gene_train_dict['gene_status_labels'] = new_gene_status.copy()
        gene_train_dict['unlikely_activators'] = unlikely_activators.copy()
        gene_train_dict['unlikely_repressors'] = unlikely_repressors.copy()
        training_dict[temp_gene] = gene_train_dict.copy()
    
    training_dict['weight_dict'] = define_weight(state_dict)
    return training_dict

def GA_fit_single_gene(training_dict, target_gene, corr_matrix, ideal_edges = 2, num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, mutation_percent_genes = 25, GA_seed = 2, init_pop_seed = 2023):
    """Identify the regulatory interactions for a gene that minimizes the discrepancy between gene states labels and simulated gene states via regulatory interactions for one single gene. 

    Args:
        training_dict (dict): The output from onesc.curate_training_data. 
        target_gene (str): The target gene. 
        corr_matrix (pandas.DataFrame): The correlation matrix 
        ideal_edges (int, optional): Ideal number of incoming edges per gene. Defaults to 2.
        num_generations (int, optional): Number of generations for genetic algorithm per gene per iteration. Defaults to 1000.
        max_iter (int, optional): Maximum number of iterations for genetic algorithm. If the fitness has not change in 3 iterations then stop early. Defaults to 10.
        num_parents_mating (int, optional): Number of parents for genetic algorithm. Defaults to 4.
        sol_per_pop (int, optional): Number of solutions to keep per generation for genetic algorithm. Defaults to 10.
        reduce_auto_reg (bool, optional): If True, remove auto activation is not needed for states satisfaction. Defaults to True.
        mutation_percent_genes (float, optional): The mutation percentage. Defaults to 25. 
        GA_seed (int, optional): The seed for genetic algorithm. Defaults to 2. 
        init_pop_seed (int, optional): The seed for generating the initial population for genetic algorithm. Defaults to 2023. 
    Returns:
        list: A pandas dataframe object containing the regulatory edges for a gene and the fitness score. 
    """
    weight_dict = training_dict['weight_dict']
    unlikely_activators = training_dict[target_gene]['unlikely_activators']
    unlikely_repressors = training_dict[target_gene]['unlikely_repressors']

    training_data = training_dict[target_gene]['feature_matrix'].copy()
    training_targets = training_dict[target_gene]['gene_status_labels'].copy()

    #ensure the correlation have the same order as compiled training data 
    corr_matrix = corr_matrix.loc[training_data.index, :]
    corr_col = corr_matrix.loc[:, target_gene]

    unlikely_activators = np.intersect1d(unlikely_activators, list(training_data.index))
    bad_activator_index = list()
    if len(unlikely_activators) > 0:
        for temp_bad_activator in unlikely_activators:
            bad_activator_index.append(np.where(training_data.index == temp_bad_activator)[0][0])

    unlikely_repressors = np.intersect1d(unlikely_repressors, list(training_data.index))
    bad_repressors_index = list()
    if len(unlikely_repressors) > 0:
        for temp_bad_repressor in unlikely_repressors:
            bad_repressors_index.append(np.where(training_data.index == temp_bad_repressor)[0][0])
        
    self_reg_index = np.where(training_data.index == target_gene)[0][0]

    def calc_activation_prob(norm_dict, upTFs): 
        if len(upTFs) == 0: 
            return 1
        elif len(upTFs) == 1: 
            return norm_dict[upTFs[0]]
        else: 
            for TF in upTFs:
                if norm_dict[TF] == 1:
                    return 1
            return 0

    def calc_repression_prob(norm_dict, downTFs): 
        if len(downTFs) == 0: 
            return 1
        elif len(downTFs) == 1: 
            return 1 - norm_dict[downTFs[0]]
        else:
            for TF in downTFs: 
                if norm_dict[TF] == 1:
                    return 0
            return 1
    
    correctness_sum = 0
    fitness_score = 0

    def max_features_fitness_func(ga_instance, solution, solution_idx):
        correctness_sum = 0
        fitness_score = 0
        # various rewards and penalties 
        additional_edge_reward = 100
        prior_edge_penalty = 2 * additional_edge_reward

        # various rewards and penalties 
        correct_reward = 1e4 * corr_matrix.shape[0]
        edge_limit_rewards = 1e3 * corr_matrix.shape[0]
        edge_limit_penalty = edge_limit_rewards / 2

        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            temp_score = int(total_prob == training_targets[i]) * correct_reward

            correctness_sum = correctness_sum + temp_score
        
        fitness_score = correctness_sum + (np.sum(np.array(solution) != 0) * additional_edge_reward)  
        
        # if the number of edges stay below ideal edge, then we add in the reward 
        if np.sum(np.abs(solution)) > ideal_edges:
            fitness_score = correctness_sum + edge_limit_penalty
        else:
            fitness_score = correctness_sum + edge_limit_rewards

        # penalize unlikely activators 
        if len(bad_activator_index) > 0: 
            for temp_index in bad_activator_index: 
                if solution[temp_index] == 1: 
                    fitness_score = fitness_score - prior_edge_penalty
        
        # penalize unlikely repressors
        if len(bad_repressors_index) > 0: 
            for temp_index in bad_repressors_index: 
                if solution[temp_index] == -1: 
                    fitness_score = fitness_score - prior_edge_penalty

        # remove self inhibition since it would not work unless we go on to protein level 
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * correct_reward)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score  
                else:
                    fitness_score = fitness_score - prior_edge_penalty # remove unnecessary auto-activator.
       
        # if the genetic algorithm direction and correlation is contradicting then -15 
        # if the genetic algoirthm direction and correlation is the same, then 10 * correlation 
        def check_direction_agreement(solution_pos, corr_pos):
            if np.sign(corr_pos) == solution_pos: 
                return True
            else: 
                return False
        
        for temp_index in list(range(0, len(corr_col))):
            if solution[temp_index] == 0:
                continue
            else:
                if check_direction_agreement(solution[temp_index], corr_col[temp_index]) == True:
                    fitness_score = fitness_score + (np.abs(corr_col[temp_index]) * 10)
                else: 
                    fitness_score = fitness_score - (np.abs(corr_col[temp_index]) * 10)
        
        if len(weight_dict) > 0:
            for temp_index in list(range(0, len(solution))):
                if solution[temp_index] == 0:
                    continue
                else:
                    if weight_dict[training_data.index[temp_index]] > (weight_dict[target_gene] + 1):
                        fitness_score = fitness_score - 1
                    else:
                        fitness_score = fitness_score + weight_dict[training_data.index[temp_index]] # this works the best 

        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - (3 * correct_reward)
        return fitness_score


    # the below are just parameters for the genetic algorithm  
    num_genes = training_data.shape[0]

    parent_selection_type = "sss"
    keep_parents = num_parents_mating
    crossover_type = "uniform"
    
    mutation_type = "random"

    # let me double check this 
    perfect_fitness = training_data.shape[1] * 1e4 * corr_matrix.shape[0] # remove the weight_dict entry

    # generate an initial population pool 
    prng = np.random.default_rng(init_pop_seed)
    init_pop_pool = np.random.choice([0, -1, 1], size=(sol_per_pop, num_genes))

    prev_1_fitness = 0 
    prev_2_fitness = 0 
    perfect_fitness_bool = False

    for run_cycle in list(range(0, max_iter)):
        fitness_function = max_features_fitness_func
        ga_instance_max = pygad.GA(num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            initial_population=init_pop_pool,
            fitness_func=fitness_function,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            parent_selection_type=parent_selection_type,
            keep_parents=keep_parents,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes, 
            suppress_warnings = True,
            gene_space = [-1, 0, 1], 
            random_seed = GA_seed)
        ga_instance_max.run()
        solution, solution_fitness, solution_idx = ga_instance_max.best_solution()
        
        init_pop_pool = ga_instance_max.population
        
        if solution_fitness >= perfect_fitness: 
            perfect_fitness_bool = True
        else:
            perfect_fitness_bool = False
        # if the fitness doesn't change for 3 rounds, then break 
        if solution_fitness == prev_1_fitness and solution_fitness == prev_2_fitness: 
            break 
        else: 
            prev_1_fitness = prev_2_fitness
            prev_2_fitness = solution_fitness

    new_edges_df = pd.DataFrame()
    
    for i in range(0, len(solution)):
        if solution[i] == 0:
            continue
        if solution[i] == -1: 
            reg_type = "-"
        elif solution[i] == 1: 
            reg_type = "+"
        temp_edge = pd.DataFrame(data = [[training_data.index[i], target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
        new_edges_df = pd.concat([new_edges_df, temp_edge])
        
    return new_edges_df

def create_network(training_dict, corr_matrix, ideal_edges = 2, num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, mutation_percent_genes = 20, GA_seed = 2, init_pop_seed = 2023): 
    """Curate a functional Boolean network using genetic algorithm that minimizes the discrepancy between gene states labels and simulated gene states via regulatory interactions. 

    Args:
        training_dict (dict): The output from onesc.curate_training_data. 
        corr_matrix (pandas.DataFrame): The Pearson correlation matrix. Could use any other distance related metrics as well. This metric is used to decide which transcription regulators to choose if there are multiple transcription regulators with the same expression states across cell states. 
        ideal_edges (int, optional): The ideal number of incoming edges per gene. Defaults to 2.
        num_generations (int, optional): Number of generations for genetic algorithm per gene per iteration. Defaults to 1000.
        max_iter (int, optional): Maximum number of iterations for genetic algorithm. If the fitness has not change in 3 iterations then stop early. Defaults to 10.
        num_parents_mating (int, optional): Number of parents for genetic algorithm. Defaults to 4.
        sol_per_pop (int, optional): Number of solutions to keep per generation for genetic algorithm. Defaults to 10.
        reduce_auto_reg (bool, optional): If True, remove auto activation is not needed for states satisfaction. Defaults to True.
        mutation_percent_genes (float, optional): The mutation percentage. Defaults to 25. 
        GA_seed (int, optional): The seed for genetic algorithm. Defaults to 2. 
        init_pop_seed (int, optional): The seed for generating the initial population for genetic algorithm. Defaults to 2023. 
    Returns:
        pandas.DataFrame: The reconstructed network. 
    """
    total_network = pd.DataFrame()
    weight_dict = training_dict['weight_dict']
    for target_gene in training_dict.keys():
        if target_gene == 'weight_dict':
            continue

        unlikely_activators = training_dict[target_gene]['unlikely_activators']
        unlikely_repressors = training_dict[target_gene]['unlikely_repressors']

        training_data = training_dict[target_gene]['feature_matrix'].copy()
        training_targets = training_dict[target_gene]['gene_status_labels'].copy()

        #ensure the correlation have the same order as compiled training data 
        corr_matrix = corr_matrix.loc[training_data.index, :]
        corr_col = corr_matrix.loc[:, target_gene]

        unlikely_activators = np.intersect1d(unlikely_activators, list(training_data.index))
        bad_activator_index = list()
        if len(unlikely_activators) > 0:
            for temp_bad_activator in unlikely_activators:
                bad_activator_index.append(np.where(training_data.index == temp_bad_activator)[0][0])

        unlikely_repressors = np.intersect1d(unlikely_repressors, list(training_data.index))
        bad_repressors_index = list()
        if len(unlikely_repressors) > 0:
            for temp_bad_repressor in unlikely_repressors:
                bad_repressors_index.append(np.where(training_data.index == temp_bad_repressor)[0][0])
            
        self_reg_index = np.where(training_data.index == target_gene)[0][0]

        def calc_activation_prob(norm_dict, upTFs): 
            if len(upTFs) == 0: 
                return 1
            elif len(upTFs) == 1: 
                return norm_dict[upTFs[0]]
            else: 
                for TF in upTFs:
                    if norm_dict[TF] == 1:
                        return 1
                return 0

        def calc_repression_prob(norm_dict, downTFs): 
            if len(downTFs) == 0: 
                return 1
            elif len(downTFs) == 1: 
                return 1 - norm_dict[downTFs[0]]
            else:
                for TF in downTFs: 
                    if norm_dict[TF] == 1:
                        return 0
                return 1

        def max_features_fitness_func(ga_instance, solution, solution_idx):
            correctness_sum = 0
            fitness_score = 0
            # various rewards and penalties 
            additional_edge_reward = 100
            prior_edge_penalty = 2 * additional_edge_reward

            # various rewards and penalties 
            correct_reward = 1e4 * corr_matrix.shape[0]
            edge_limit_rewards = 1e3 * corr_matrix.shape[0]
            edge_limit_penalty = edge_limit_rewards / 2

            for i in range(0, len(training_data.columns)):
                state = training_data.columns[i]
                norm_dict = training_data.loc[:, state]
                upTFs = training_data.index[np.array(solution) == 1]
                downTFs = training_data.index[np.array(solution) == -1]
                activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
                repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

                total_prob = activation_prob * repression_prob

                temp_score = int(total_prob == training_targets[i]) * correct_reward

                correctness_sum = correctness_sum + temp_score
            
            fitness_score = correctness_sum + (np.sum(np.array(solution) != 0) * additional_edge_reward)  # if a gene can be either activator or inhibitor, choose inhibitor
            
            # if the number of edges stay below ideal edge, then we add in the reward 
            if np.sum(np.abs(solution)) > ideal_edges:
                fitness_score = fitness_score + edge_limit_penalty
                #fitness_score = fitness_score - ((np.sum(np.abs(solution)) - ideal_edges) * 2 * additional_edge_reward)
            else:
                fitness_score = fitness_score + edge_limit_rewards

            # penalize unlikely activators 
            if len(bad_activator_index) > 0: 
                for temp_index in bad_activator_index: 
                    if solution[temp_index] == 1: 
                        fitness_score = fitness_score - prior_edge_penalty
            
            # penalize unlikely repressors
            if len(bad_repressors_index) > 0: 
                for temp_index in bad_repressors_index: 
                    if solution[temp_index] == -1: 
                        fitness_score = fitness_score - prior_edge_penalty

            # remove self inhibition since it would not work
            if self_reg_index > -1:
                if solution[self_reg_index] == -1:
                    fitness_score = fitness_score - (3 * correct_reward)
                elif solution[self_reg_index] == 1:
                    if reduce_auto_reg == False:
                        fitness_score = fitness_score  
                    else:
                        fitness_score = fitness_score - prior_edge_penalty # remove unnecessary auto-activator.
        
            # if the genetic algorithm direction and correlation is contradicting then -10 * correlation 
            # if the genetic algoirthm direction and correlation is the same, then 10 * correlation 
            def check_direction_agreement(solution_pos, corr_pos):
                if np.sign(corr_pos) == solution_pos: 
                    return True
                else: 
                    return False
            
            for temp_index in list(range(0, len(corr_col))):
                if solution[temp_index] == 0:
                    continue
                else:
                    if check_direction_agreement(solution[temp_index], corr_col[temp_index]) == True:
                        fitness_score = fitness_score + (np.abs(corr_col[temp_index]) * 10)
                    else: 
                        fitness_score = fitness_score - (np.abs(corr_col[temp_index]) * 10)
            
            if len(weight_dict) > 0:
                for temp_index in list(range(0, len(solution))):
                    if solution[temp_index] == 0:
                        continue
                    else:
                        if weight_dict[training_data.index[temp_index]] > (weight_dict[target_gene] + 1):
                            fitness_score = fitness_score - 1
                        else:
                            fitness_score = fitness_score + weight_dict[training_data.index[temp_index]]  
            if np.sum(np.abs(solution)) == 0: # prevent cases of a target gene without any regulations including self-regulations. 
                fitness_score = fitness_score - (3 * correct_reward)
            return fitness_score

        # the below are just parameters for the genetic algorithm  
        num_genes = training_data.shape[0]

        parent_selection_type = "sss"
        keep_parents = num_parents_mating
        crossover_type = "uniform"
        
        mutation_type = "random"

        # let me double check this 
        perfect_fitness = training_data.shape[1] * 1e4 * corr_matrix.shape[0] # remove the weight_dict entry

        # generate an initial population pool 
        prng = np.random.default_rng(init_pop_seed)
        init_pop_pool = prng.choice([0, -1, 1], size=(sol_per_pop, num_genes))

        prev_1_fitness = 0 
        prev_2_fitness = 0 
        perfect_fitness_bool = False

        for run_cycle in list(range(0, max_iter)):
            fitness_function = max_features_fitness_func
            ga_instance_max = pygad.GA(num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                initial_population=init_pop_pool,
                fitness_func=fitness_function,
                sol_per_pop=sol_per_pop,
                num_genes=num_genes,
                parent_selection_type=parent_selection_type,
                keep_parents=keep_parents,
                crossover_type=crossover_type,
                mutation_type=mutation_type,
                mutation_percent_genes=mutation_percent_genes, 
                suppress_warnings = True,
                gene_space = [-1, 0, 1], 
                random_seed = GA_seed)
            ga_instance_max.run()
            solution, solution_fitness, solution_idx = ga_instance_max.best_solution()
            
            init_pop_pool = ga_instance_max.population
            
            if solution_fitness >= perfect_fitness: 
                perfect_fitness_bool = True
            else:
                perfect_fitness_bool = False
            # if the fitness doesn't change for 3 rounds, then break 
            if solution_fitness == prev_1_fitness and solution_fitness == prev_2_fitness: 
                break 
            else: 
                prev_1_fitness = prev_2_fitness
                prev_2_fitness = solution_fitness

        new_edges_df = pd.DataFrame()
        
        for i in range(0, len(solution)):
            if solution[i] == 0:
                continue
            if solution[i] == -1: 
                reg_type = "-"
            elif solution[i] == 1: 
                reg_type = "+"
            temp_edge = pd.DataFrame(data = [[training_data.index[i], target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
            new_edges_df = pd.concat([new_edges_df, temp_edge])

        total_network = pd.concat([total_network, new_edges_df])
        '''
        if perfect_fitness_bool == False: 
            print(target_gene + " does not fit perfectly")
        else:
            print(target_gene + " finished fitting")
        '''
    return total_network

def create_network_ensemble(training_dict, corr_matrix, n_cores = 16, run_parallel = True, ideal_edges = 2, num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, mutation_percent_genes = 20, GA_seed_list = [1, 2, 3, 4], init_pop_seed_list = [21, 22, 23, 24], **kwargs):
    """Create an ensemble of inferred networks using different genetic algorithm and initial population seeds. 

    Args:
        training_dict (dict): The output from onesc.curate_training_data. 
        corr_matrix (pandas.DataFrame): The Pearson correlation matrix. Could use any other distance related metrics as well as long as they are between 0 and 1. This metric is used to decide which transcription regulators to choose if there are multiple transcription regulators with the same expression states across cell states. 
        n_cores (int, optional): number of cores to run the network inference in parallel. Defaults to 16.
        run_parallel (bool, optional): whether to run network inference in parallel. Defaults to True.
        ideal_edges (int, optional): The ideal number of incoming edges per gene. Defaults to 2.
        num_generations (int, optional): Number of generations for genetic algorithm per gene per iteration. Defaults to 1000.
        max_iter (int, optional): Maximum number of iterations for genetic algorithm. If the fitness has not change in 3 iterations then stop early. Defaults to 10.
        num_parents_mating (int, optional): Number of parents for genetic algorithm. Defaults to 4.
        sol_per_pop (int, optional): Number of solutions to keep per generation for genetic algorithm. Defaults to 10.
        reduce_auto_reg (bool, optional): If True, remove auto activation is not needed for states satisfaction. Defaults to True.
        mutation_percent_genes (float, optional): The mutation percentage. Defaults to 25. 
        GA_seed_list (list, optional): a list of seeds for genetic algorithm. Defaults to [1, 2, 3, 4].
        init_pop_seed_list (list, optional): a list of seeds for generating initial populations. Defaults to [20, 21, 23, 24].

    Returns:
        _type_: _description_
    """
    seeds_combo = list(itertools.product(GA_seed_list, init_pop_seed_list))

    def mass_train(temp_GA_seed, temp_init_seed):
        inferred_network = create_network(training_dict, 
                                          corr_matrix, 
                                          ideal_edges, 
                                          num_generations, 
                                          max_iter, 
                                          num_parents_mating, 
                                          sol_per_pop, 
                                          reduce_auto_reg, 
                                          mutation_percent_genes,
                                          GA_seed = temp_GA_seed, 
                                          init_pop_seed = temp_init_seed)
        return inferred_network

    if n_cores > cpu_count(): 
        warnings.warn("Maximum number of cores is " + str(cpu_count()))
        n_cores = cpu_count()
    if run_parallel == True: 
        parallel_results = Parallel(n_jobs=n_cores)(
            delayed(mass_train)(*args) for args in seeds_combo
        )
    else: 
        parallel_results = list()
        for temp_GA_seed, temp_init_seed in seeds_combo:
            parallel_results.append(mass_train(temp_GA_seed, temp_init_seed))

    def get_majority_network(parallel_results):
        max_network = pd.DataFrame()
        for temp_gene in corr_matrix.index: 
            stats_df = pd.DataFrame()
            temp_subnet_dict = dict()
            for i in range(0, len(parallel_results)):
                temp_subnet = parallel_results[i].loc[parallel_results[i]['TG'] == temp_gene, :].copy()
                temp_subnet = temp_subnet.sort_values(by='TF', ascending=True)
                if temp_subnet.to_string() not in stats_df.index:
                    stats_dict = {'occurrences': [1], 'edge_diff': np.abs(temp_subnet.shape[0] - ideal_edges)}
                    row_df = pd.DataFrame(stats_dict)
                    row_df.index = [temp_subnet.to_string()]
                    stats_df = pd.concat([stats_df, row_df])
                    temp_subnet_dict[temp_subnet.to_string()] = temp_subnet
                else: 
                    stats_df.loc[temp_subnet.to_string(), 'occurrences'] = stats_df.loc[temp_subnet.to_string(), 'occurrences'] + 1
            filtered_stats_df = stats_df.loc[stats_df['occurrences'] == np.max(stats_df['occurrences']), :].copy()
            filtered_stats_df = filtered_stats_df.sort_values('edge_diff', ascending = True)

            max_network = pd.concat([max_network, temp_subnet_dict[filtered_stats_df.index[0]]])
        return max_network

    majority_network = get_majority_network(parallel_results)

    return [majority_network, parallel_results]

def calc_corr(train_exp):
    return train_exp.T.corr()

def define_weight(state_dict):
    """Define the rank weight of the transcription regulators. In this function, the earlier along the trajectory a gene is activated, the higher the weight (more likely to get assigned to regulate other genes). 
    The users can design their own weight dictionary if they want. As long as the format is consistent with the output of this function. 

    Args:
        state_dict (dict): The output from onesc.define_states. It is a dictionary containing all the cell states at which genes transition from ON to OFF or OFF to ON. 

    Returns:
        dict: A dictionary of weights that are related to how early does the gene become active for all the genes. 
    """
    weight_dict = dict()
    max_len = 0
    all_genes = state_dict[list(state_dict.keys())[0]].index
    for traj in state_dict.keys():
        temp_state = state_dict[traj]
        if temp_state.shape[1] > max_len:
            max_len = temp_state.shape[1]
        for temp_col_index in range(0, temp_state.shape[1]):
            active_genes_list = temp_state.index[temp_state.iloc[:, temp_col_index] == 1]
            for active_gene in active_genes_list:
                if active_gene in weight_dict.keys():
                    if temp_col_index < weight_dict[active_gene]:
                        weight_dict[active_gene] = temp_col_index
                else:
                    weight_dict[active_gene] = temp_col_index
    for temp_gene in all_genes:
        if temp_gene in weight_dict.keys():
            weight_dict[temp_gene] = max_len - weight_dict[temp_gene]
        else: 
            weight_dict[temp_gene] = 0
    return weight_dict