import numpy as np 
import pandas as pd
import pygad

def define_states(exp_tab, samp_tab, lineage_cluster, vector_thresh, cluster_col = 'cluster_id'):
    """Define booleanized and discrete cell state profiles 

    Args:
        exp_tab (pandas.DataFrame): scRNA-seq expression matrix. Rownames are genes; column names are sample barcodes. 
        samp_tab (pandas.DataFrame): sample Ta
        lineage_cluster (_type_): _description_
        vector_thresh (_type_): _description_
        cluster_col (str, optional): _description_. Defaults to 'cluster_id'.

    Returns:
        _type_: _description_
    """
    state_dict = dict()
    for lineage in lineage_cluster.keys():
        
        temp_df = pd.DataFrame()
        for cell_type in lineage_cluster[lineage]:
            sub_st = samp_tab.loc[samp_tab[cluster_col] == cell_type, :]
            sub_exp = exp_tab.loc[:, sub_st.index]
            temp_df[cell_type] = (sub_exp.mean(axis = 1) >= vector_thresh) * 1
        state_dict[lineage] = temp_df
    return state_dict  

def define_transition(state_dict):
    transition_dict = dict()
    for lineage in state_dict.keys():
        temp_df = pd.DataFrame()
        initial_condition = True
        prev_col = None 
        for cell_type in state_dict[lineage].columns:
            if initial_condition == True: 
                temp_trans = state_dict[lineage][cell_type]
                temp_df[cell_type] = temp_trans.replace(0, -1)
                prev_col = cell_type
                initial_condition = False
            else: 
                temp_trans = state_dict[lineage][cell_type] - state_dict[lineage][prev_col]
                temp_df[cell_type] = temp_trans
                prev_col = cell_type
                
        transition_dict[lineage] = temp_df
    return transition_dict

def curate_training_data(state_dict, transition_dict, lineage_time_change_dict, samp_tab, cluster_id = "leiden", pt_id = "dpt_pseudotime",act_tolerance = 0.01, selected_regulators = list()):
    all_genes = transition_dict['lineage_0'].index

    def extract_steady_states(state_df, target_gene, lineage_name):
        extract_ss_df = state_df.iloc[:, -1:].copy()
        extract_ss_df.columns = extract_ss_df.columns + "_SS_" + lineage_name
        target_gene_state = [extract_ss_df.loc[target_gene, extract_ss_df.columns[0]]]
        prior_df = state_df.iloc[:, -2]
        if prior_df[target_gene] == 1:
            extract_ss_df.loc[target_gene, extract_ss_df.columns[0]] = 1
        else:
            extract_ss_df.loc[target_gene, extract_ss_df.columns[0]] = 0
        return [target_gene_state, extract_ss_df]

    def extract_stable_state(state_df, target_gene, unstable_states_list, lineage_name):
        if len(unstable_states_list) == 0:
            extract_stable_df = state_df.drop(state_df.columns[np.array([0, state_df.shape[1] - 1])], axis = 1).copy()
        else:
            exclude_index_list = np.array([0])
            exclude_index_list = np.append(exclude_index_list, state_df.shape[1] - 1)
            exclude_index_list = np.unique(exclude_index_list)
            extract_stable_df = state_df.drop(state_df.columns[exclude_index_list], axis = 1).copy()
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
        extract_stable_df.columns = extract_stable_df.columns + "_stable_" + lineage_name
        return [target_gene_state, extract_stable_df]

    def check_stable_initial(trans_dict, target_gene):
        for temp_lineage in trans_dict.keys():
            temp_transition = trans_dict[temp_lineage]
            if 0 in np.where(temp_transition.loc[target_gene, :] != 0)[0] - 1:
                return False
        return True

    def extract_stable_initial_state(state_dict, target_gene, initial_stability = True):
        if initial_stability == True:
            temp_state = state_dict['lineage_0']
            extract_ss_initial_df = temp_state.iloc[:, 0:1].copy()
            extract_ss_initial_df.columns = extract_ss_initial_df.columns + "_initial_SS_all"
            target_gene_state = [extract_ss_initial_df.loc[target_gene, extract_ss_initial_df.columns[0]]]
            return [target_gene_state, extract_ss_initial_df]
        else:
            return [[], pd.DataFrame()]

    def extract_unstable_state(state_df, transition_df, time_change_df, target_gene, exclude_index_list, samp_tab, cluster_id, pt_id, lineage_name, act_tolerance = 0.01):
        target_gene_state = []
        extract_unstable_df = pd.DataFrame()

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
            return [[], pd.DataFrame(), []]
        for temp_index in exclude_index_list: 
            target_gene_state.append(state_df.iloc[:, temp_index + 1][target_gene])
            temp_extract_unstable_df = state_df.iloc[:, temp_index:temp_index + 1].copy()
            potential_regulators = find_potential_regulators(transition_df, time_change_df, target_gene, temp_index, cluster_id, pt_id, act_tolerance)
            for potential_regulator in potential_regulators:
                temp_extract_unstable_df.loc[potential_regulator, temp_extract_unstable_df.columns[0]] = state_df.loc[potential_regulator, state_df.columns[temp_index + 1]]

            extract_unstable_df = pd.concat([extract_unstable_df, temp_extract_unstable_df], axis = 1)
        
        # mark down the states that is unstable in any of the trajectory 
        if len(extract_unstable_df.columns) > 0: 
            unstable_states = list(extract_unstable_df.columns)
        else: 
            unstable_states = []
        extract_unstable_df.columns = extract_unstable_df.columns + "_unstable_" + lineage_name 
        return [target_gene_state, extract_unstable_df, unstable_states]

    training_dict = dict()
    for temp_gene in all_genes: 

        gene_status_label = []
        feature_mat = pd.DataFrame()
        gene_train_dict = dict()

        # get initial state if it is stable for the gene
        stable_initial = check_stable_initial(transition_dict, temp_gene)
        [temp_label, temp_feature] = extract_stable_initial_state(state_dict, temp_gene, stable_initial)
        gene_status_label = gene_status_label + temp_label
        feature_mat = pd.concat([feature_mat, temp_feature], axis = 1).copy()

        # grab all the unstable states 
        unstable_states_list = []
        for temp_lineage in transition_dict.keys():
            temp_transition = transition_dict[temp_lineage] # transition matrix 
            temp_state = state_dict[temp_lineage] # state matrix 
            temp_time_change = lineage_time_change_dict[temp_lineage] # time series 
            
            # find the index of the state before transition of temp_gene 
            # the states that were right before a transition is not stable 
            prev_index_list = np.where(temp_transition.loc[temp_gene, :] != 0)[0] - 1
            prev_index_list = prev_index_list[prev_index_list > -1]
        
            # get the unstable states
            returned_list = extract_unstable_state(temp_state, temp_transition, temp_time_change, temp_gene, prev_index_list, samp_tab, cluster_id, pt_id, temp_lineage, act_tolerance)
            temp_label = returned_list[0]
            temp_feature = returned_list[1]
            unstable_states = returned_list[2]
            gene_status_label = gene_status_label + temp_label
            feature_mat = pd.concat([feature_mat, temp_feature], axis = 1).copy()
            unstable_states_list = unstable_states_list + unstable_states
        
        # list of states that are unstable across all trajectories 
        unstable_states_list = np.unique(unstable_states_list)

        for temp_lineage in transition_dict.keys():
            temp_state = state_dict[temp_lineage] # state matrix 
            
            # get the steady states 
            [temp_label, temp_feature] = extract_steady_states(temp_state, temp_gene, temp_lineage)
            gene_status_label = gene_status_label + temp_label
            feature_mat = pd.concat([feature_mat, temp_feature], axis = 1).copy()

            # get stable states. aka genes in states that do not change immediately 
            # TODO add an argument of all the unstable states so we can avoid them even if they are in a different lineage  
            [temp_label, temp_feature] = extract_stable_state(temp_state, temp_gene, unstable_states_list, temp_lineage)
            gene_status_label = gene_status_label + temp_label
            feature_mat = pd.concat([feature_mat, temp_feature], axis = 1).copy()
        
        if len(selected_regulators) != 0:
            feature_mat = feature_mat.loc[selected_regulators, :]
        gene_train_dict['feature_matrix'] = feature_mat.copy()
        gene_train_dict['gene_status_labels'] = gene_status_label.copy()
        training_dict[temp_gene] = gene_train_dict.copy()
    return training_dict

def GA_fit_data(training_dict, target_gene, selected_regulators = list(), regulators_rank = list(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, max_edge_first = False, max_dup_genes = 2): 
    training_data = training_dict[target_gene]['feature_matrix']
    if len(selected_regulators) > 0: 
        selected_regulators = np.append(selected_regulators, target_gene)
        selected_regulators = np.unique(selected_regulators)
        training_data = training_data.loc[selected_regulators, :]

    training_data_original = training_data.copy()

    training_data = training_data.loc[training_data.sum(axis = 1) > 0, :] # remove genes that are not active in any states 
    training_data = training_data.drop_duplicates() # remove duplicated genes with the same state profiles 
    
    training_targets = training_dict[target_gene]['gene_status_labels']
    feature_dict = dict()

    def to_string_pattern(gene_pattern):
        gene_pattern = [str(x) for x in gene_pattern]
        return "_".join(gene_pattern)

    for regulator in training_data_original.index: 
        gene_pattern = training_data_original.loc[regulator, :]
        x_str = to_string_pattern(gene_pattern)
        if x_str in feature_dict.keys():
            feature_dict[x_str] = np.concatenate([feature_dict[x_str], [regulator]])
        else:
            feature_dict[x_str] = np.array([regulator])
    
    # rename the training data 
    training_data.index = training_data.apply(to_string_pattern, axis = 1)
    target_gene_pattern = to_string_pattern(training_data_original.loc[target_gene, :]) # get the TG self activation gene profile

    # the sole purpose of the code block below is to isolate the self-regulation out so that if there is a self-inhibiton, we will penalize that 
    # or if we want to reduce self-activation, we can penalize that as well. The key is to find out where the index of the self-activation 
    if target_gene_pattern in training_data.index: # if the self-activation profile exists in the training data...it really should unless it was consistently 0 across the board 
        # if the self regulation pattern match with 1 or more other gene profile, then we isolate out self activation 
        # in this case, it that self activation happens to be self-inhibition, then we penalize that  
        if len(feature_dict[target_gene_pattern]) > 1: # if the there are more than at least 1 other gene pattern that is the same as the self regulation
            feature_dict[target_gene_pattern] = feature_dict[target_gene_pattern][feature_dict[target_gene_pattern] != target_gene]
            feature_dict[target_gene + "_" + target_gene_pattern] = [target_gene]
            target_gene_df = training_data_original.loc[target_gene, :].to_frame().T
            target_gene_df.index = [target_gene + "_" + target_gene_pattern]
            training_data = pd.concat([training_data, target_gene_df])
        else:
            feature_dict.pop(target_gene_pattern)
            feature_dict[target_gene + "_" + target_gene_pattern] = [target_gene]
            training_data = training_data.rename(index={target_gene_pattern:target_gene + "_" + target_gene_pattern})
        # find where the index of self regulation
        self_reg_index = np.where(training_data.index == target_gene + "_" + target_gene_pattern)[0][0]
    else:
        self_reg_index = -1 # if for whatever reason 

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
    
    def max_features_fitness_func(solution, solution_idx):
        correctness_sum = 0

        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            temp_score = int(total_prob == training_targets[i]) * 1000

            correctness_sum = correctness_sum + temp_score
        fitness_score = correctness_sum + (np.sum(solution == 1) * 10) + (np.sum(solution == -1) * 15) # if a gene can be either activator or inhibitor, choose inhibitor

        # remove self inhibition since it would not work unless we go on to protein level 
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * 1000)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score  
                else:
                    fitness_score = fitness_score - 15 # remove unnecessary auto-activator.
        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - (3 * 1000)
        return fitness_score

    def min_features_fitness_func(solution, solution_idx):
        correctness_sum = 0
        
        correct_scale = 1000
        activated_bonus_scale = 0.001
        
        # calculate the agreement of the network confirguation with real data
        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            temp_score = int(total_prob == training_targets[i]) * correct_scale
            correctness_sum = correctness_sum + temp_score

        #fitness_score = correctness_sum + (np.sum(solution == 0) * 1) + (np.sum(solution == -1) * 0.1) #if an edge can be either activator or inhibitor, choose activator 
        fitness_score = correctness_sum + (np.sum(solution == 0) * 10) #minimize the number of edges 

        # penalize the self inhibitors
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * correct_scale)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score + 1 # add to the fitness to add more self-regulation
                else:
                    fitness_score = fitness_score
        fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) != 0, :].sum()) * activated_bonus_scale # favor genes that are turned on across more states 
        #fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) == -1, :].sum()) * activated_bonus_scale # favor repressive edges  

        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - 3000
        return fitness_score

    # the below are just parameters for the genetic algorithm  

    num_genes = training_data.shape[0]

    parent_selection_type = "sss"
    keep_parents = num_parents_mating
    crossover_type = "uniform"
    
    mutation_type = "random"
    mutation_percent_genes = 10
    #mutation_percent_genes = [40, 10] # maybe try to use number of genes as 

    perfect_fitness = training_data.shape[1] * 1000

    # generate an initial population pool 
    init_pop_pool = np.random.choice([0, -1, 1], size=(sol_per_pop, num_genes))

    prev_1_fitness = 0 
    prev_2_fitness = 0 
    perfect_fitness_bool = False

    for run_cycle in list(range(0, max_iter)):
        fitness_function = min_features_fitness_func
        ga_instance_min = pygad.GA(num_generations=num_generations,
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
                        gene_space = [-1, 0, 1])
        ga_instance_min.run()
        first_solution, first_solution_fitness, first_solution_idx = ga_instance_min.best_solution()
        
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
            gene_space = [-1, 0, 1])
        ga_instance_max.run()
        second_solution, second_solution_fitness, second_solution_idx = ga_instance_max.best_solution()
        
        # If both solutions would output perfect reachability, then pick the one preferred by the user
        if second_solution_fitness >= perfect_fitness and first_solution_fitness >= perfect_fitness:
            if max_edge_first == False: 
                solution = first_solution 
                solution_fitness = first_solution_fitness
                init_pop_pool = ga_instance_min.population
            else:
                solution = second_solution
                solution_fitness = second_solution_fitness
                init_pop_pool = ga_instance_max.population
        else: 
            if second_solution_fitness > first_solution_fitness:  
                solution = second_solution
                solution_fitness = second_solution_fitness
                init_pop_pool = ga_instance_max.population
            else:
                solution = first_solution 
                solution_fitness = first_solution_fitness
                init_pop_pool = ga_instance_min.population
        
        if solution_fitness >= perfect_fitness: 
            perfect_fitness_bool = True

        # if the fitness doesn't change for 3 rounds, then break 
        if solution_fitness == prev_1_fitness and solution_fitness == prev_2_fitness: 
            break 
        else: 
            prev_1_fitness = prev_2_fitness
            prev_2_fitness = solution_fitness

    new_edges_df = pd.DataFrame()
    
    for i in range(0, training_data.shape[0]):
        if solution[i] == 0:
            continue

        x_str = training_data.index[i]

        if solution[i] == -1: 
            reg_type = "-"
        elif solution[i] == 1: 
            reg_type = "+"
        
        # This is where I would select the top regulator for each state 
        if len(regulators_rank) == 0: 
            for regulator in feature_dict[x_str]:
                temp_edge = pd.DataFrame(data = [[regulator, target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
                new_edges_df = pd.concat([new_edges_df, temp_edge])
        else: #TODO make this a little bit more elegant. Maybe instead of using correlation, find other metrics 
            dup_genes = feature_dict[x_str]
            reg_rank = np.array(range(0, len(regulators_rank)))
            intersect_index = np.where(np.isin(regulators_rank, dup_genes))

            intersect_genes = regulators_rank[intersect_index]
            reg_rank = reg_rank[intersect_index]
            intersect_genes = intersect_genes[np.argsort(reg_rank)]

            if len(intersect_genes) > max_dup_genes:
                intersect_genes = intersect_genes[0:max_dup_genes]

            for regulator in intersect_genes:
                temp_edge = pd.DataFrame(data = [[regulator, target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
                new_edges_df = pd.concat([new_edges_df, temp_edge])
            
    return [new_edges_df, perfect_fitness_bool]

def create_network(training_dict, selected_regulators_dict = dict(), regulators_rank_dict = dict(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, max_edge_first = False, max_dup_genes = 2): 
    total_network = pd.DataFrame()
    for temp_gene in training_dict.keys():
        if temp_gene in selected_regulators_dict.keys():
            selected_regulators_list = selected_regulators_dict[temp_gene]
        else:
            selected_regulators_list = list()

        if temp_gene in regulators_rank_dict.keys():
            regulators_rank_list = regulators_rank_dict[temp_gene]
        else:
            regulators_rank_list = list()

        new_network, perfect_fitness_bool = GA_fit_data(training_dict, 
                                                        temp_gene, 
                                                        selected_regulators = selected_regulators_list, 
                                                        regulators_rank = regulators_rank_list,
                                                        num_generations = num_generations, 
                                                        max_iter = max_iter, 
                                                        num_parents_mating = num_parents_mating, 
                                                        sol_per_pop = sol_per_pop, 
                                                        reduce_auto_reg = reduce_auto_reg, 
                                                        max_edge_first = max_edge_first, 
                                                        max_dup_genes = max_dup_genes)
        if perfect_fitness_bool == False:
            new_network, perfect_fitness_bool = GA_fit_data(training_dict, 
                                            temp_gene, 
                                            regulators_rank = regulators_rank_list,
                                            num_generations = num_generations, 
                                            max_iter = max_iter, 
                                            num_parents_mating = num_parents_mating, 
                                            sol_per_pop = sol_per_pop, 
                                            reduce_auto_reg = reduce_auto_reg, 
                                            max_edge_first = max_edge_first, 
                                            max_dup_genes = max_dup_genes)
        total_network = pd.concat([total_network, new_network])
        if perfect_fitness_bool == False: 
            print(temp_gene + " does not fit perfectly")
    return total_network

def create_network_serial(training_dict, regulators_rank_dict = dict(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, max_edge_first = False, max_dup_genes = 2):
    total_network = pd.DataFrame()
    for temp_gene in training_dict.keys():

        if temp_gene in regulators_rank_dict.keys(): 
            rank_regulators_list = regulators_rank_dict[temp_gene]
        else:
            rank_regulators_list = list()

        if len(rank_regulators_list) < 4: 
            rank_list = [len(rank_regulators_list)]
        else: 
            rank_list = list(range(4, len(rank_regulators_list) + 1))
        for end_index in rank_list:
            sub_rank_reg_list = rank_regulators_list[0:end_index]
            new_network, perfect_fitness_bool = GA_fit_data(training_dict, 
                                                            temp_gene, 
                                                            selected_regulators = sub_rank_reg_list, 
                                                            num_generations = num_generations, 
                                                            max_iter = max_iter, 
                                                            num_parents_mating = num_parents_mating, 
                                                            sol_per_pop = sol_per_pop, 
                                                            reduce_auto_reg = reduce_auto_reg, 
                                                            max_edge_first = max_edge_first, 
                                                            max_dup_genes = max_dup_genes)
            if perfect_fitness_bool == True: 
                break 
        total_network = pd.concat([total_network, new_network])
    return total_network

def create_network_iter(training_dict, initial_state, regulators_rank_dict = dict(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, remove_bad_genes = False, max_edge_first = False, max_dup_genes = 2): 
    total_network = pd.DataFrame()
    for temp_gene in training_dict.keys():

        if temp_gene in regulators_rank_dict.keys(): 
            rank_regulators_list = regulators_rank_dict[temp_gene]
        else:
            rank_regulators_list = list()

        if len(rank_regulators_list) < 3: 
            rank_list = [len(rank_regulators_list)]
        else: 
            rank_list = list(range(3, len(rank_regulators_list)))
        for end_index in rank_list:
            sub_rank_reg_list = rank_regulators_list[0:end_index]
            new_network, perfect_fitness_bool = GA_fit_data(training_dict, 
                                                            temp_gene, 
                                                            initial_state, 
                                                            selected_regulators = sub_rank_reg_list, 
                                                            num_generations = num_generations, 
                                                            max_iter = max_iter, 
                                                            num_parents_mating = num_parents_mating, 
                                                            sol_per_pop = sol_per_pop, 
                                                            reduce_auto_reg = reduce_auto_reg, 
                                                            remove_bad_genes = remove_bad_genes, 
                                                            max_edge_first = max_edge_first, 
                                                            max_dup_genes = max_dup_genes)
            if perfect_fitness_bool == True: 
                break 

        if perfect_fitness_bool == False:
            new_network, perfect_fitness_bool = GA_fit_data(training_dict, 
                                            temp_gene, 
                                            initial_state, 
                                            num_generations = num_generations, 
                                            max_iter = max_iter, 
                                            num_parents_mating = num_parents_mating, 
                                            sol_per_pop = sol_per_pop, 
                                            reduce_auto_reg = reduce_auto_reg, 
                                            remove_bad_genes = remove_bad_genes, 
                                            max_edge_first = max_edge_first, 
                                            max_dup_genes = max_dup_genes)

        total_network = pd.concat([total_network, new_network])
    return total_network

def GA_fit_data_penalize(training_dict, target_gene, selected_regulators = list(), regulators_rank = list(), unlikely_activators = list(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, max_edge_first = False, max_dup_genes = 2): 
    training_data = training_dict[target_gene]['feature_matrix']
    if len(selected_regulators) > 0: 
        selected_regulators = np.append(selected_regulators, target_gene)
        selected_regulators = np.unique(selected_regulators)
        training_data = training_data.loc[selected_regulators, :]

    training_data_original = training_data.copy()

    training_data = training_data.loc[training_data.sum(axis = 1) > 0, :] # remove genes that are not active in any states 
    training_data = training_data.drop_duplicates() # remove duplicated genes with the same state profiles 
    
    training_targets = training_dict[target_gene]['gene_status_labels']
    feature_dict = dict()
    rev_feature_dict = dict()

    def to_string_pattern(gene_pattern):
        gene_pattern = [str(x) for x in gene_pattern]
        return "_".join(gene_pattern)

    for regulator in training_data_original.index: 
        gene_pattern = training_data_original.loc[regulator, :]
        x_str = to_string_pattern(gene_pattern)
        if x_str in feature_dict.keys():
            feature_dict[x_str] = np.concatenate([feature_dict[x_str], [regulator]])
            rev_feature_dict[regulator] = x_str
        else:
            feature_dict[x_str] = np.array([regulator])
            rev_feature_dict[regulator] = x_str

    # rename the training data 
    training_data.index = training_data.apply(to_string_pattern, axis = 1)
    target_gene_pattern = to_string_pattern(training_data_original.loc[target_gene, :]) # get the TG self activation gene profile

    # the sole purpose of the code block below is to isolate the self-regulation out so that if there is a self-inhibiton, we will penalize that 
    # or if we want to reduce self-activation, we can penalize that as well. The key is to find out where the index of the self-activation 
    if target_gene_pattern in training_data.index: # if the self-activation profile exists in the training data...it really should unless it was consistently 0 across the board 
        # if the self regulation pattern match with 1 or more other gene profile, then we isolate out self activation 
        # in this case, it that self activation happens to be self-inhibition, then we penalize that  
        if len(feature_dict[target_gene_pattern]) > 1: # if the there are more than at least 1 other gene pattern that is the same as the self regulation
            feature_dict[target_gene_pattern] = feature_dict[target_gene_pattern][feature_dict[target_gene_pattern] != target_gene]
            feature_dict[target_gene + "_" + target_gene_pattern] = [target_gene]
            target_gene_df = training_data_original.loc[target_gene, :].to_frame().T
            target_gene_df.index = [target_gene + "_" + target_gene_pattern]
            training_data = pd.concat([training_data, target_gene_df])
        else:
            feature_dict.pop(target_gene_pattern)
            feature_dict[target_gene + "_" + target_gene_pattern] = [target_gene]
            training_data = training_data.rename(index={target_gene_pattern:target_gene + "_" + target_gene_pattern})
        # find where the index of self regulation
        self_reg_index = np.where(training_data.index == target_gene + "_" + target_gene_pattern)[0][0]
    else:
        self_reg_index = -1 # if for whatever reason 

    # figure out the index at which bad activators occur 
    bad_activator_index = list()
    if len(unlikely_activators) > 0:
        for temp_bad_activator in unlikely_activators:
            str_pattern = rev_feature_dict[temp_bad_activator]
            bad_activator_index.append(np.where(training_data.index == str_pattern)[0][0])
        bad_activator_index = np.unique(bad_activator_index)

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
    
    def max_features_fitness_func(solution, solution_idx):
        correctness_sum = 0

        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            temp_score = int(total_prob == training_targets[i]) * 1000

            correctness_sum = correctness_sum + temp_score
        fitness_score = correctness_sum + (np.sum(solution == 1) * 10) + (np.sum(solution == -1) * 15) # if a gene can be either activator or inhibitor, choose inhibitor
        
        # penalize unlikely activators 
        if len(bad_activator_index) > 0: 
            for temp_index in bad_activator_index: 
                if solution[temp_index] == 1: 
                    fitness_score = fitness_score - 40

        # remove self inhibition since it would not work unless we go on to protein level 
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * 1000)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score  
                else:
                    fitness_score = fitness_score - 15 # remove unnecessary auto-activator.
        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - (3 * 1000)
        return fitness_score

    def min_features_fitness_func(solution, solution_idx):
        correctness_sum = 0
        
        correct_scale = 1000
        activated_bonus_scale = 0.001
        
        # calculate the agreement of the network confirguation with real data
        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            temp_score = int(total_prob == training_targets[i]) * correct_scale
            correctness_sum = correctness_sum + temp_score

        #fitness_score = correctness_sum + (np.sum(solution == 0) * 1) + (np.sum(solution == -1) * 0.1) #if an edge can be either activator or inhibitor, choose activator 
        fitness_score = correctness_sum + (np.sum(solution == 0) * 10) #minimize the number of edges 

        # penalize unlikely activators 
        if len(bad_activator_index) > 0: 
            for temp_index in bad_activator_index: 
                if solution[temp_index] == 1: 
                    fitness_score = fitness_score - 40
        
        # penalize the self inhibitors
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * correct_scale)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score + 1 # add to the fitness to add more self-regulation
                else:
                    fitness_score = fitness_score
        fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) != 0, :].sum()) * activated_bonus_scale # favor genes that are turned on across more states 
        #fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) == -1, :].sum()) * activated_bonus_scale # favor repressive edges  
        
        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - 3000
        return fitness_score

    # the below are just parameters for the genetic algorithm  

    num_genes = training_data.shape[0]

    parent_selection_type = "sss"
    keep_parents = num_parents_mating
    crossover_type = "uniform"
    
    mutation_type = "random"
    mutation_percent_genes = 10
    #mutation_percent_genes = [40, 10] # maybe try to use number of genes as 

    perfect_fitness = training_data.shape[1] * 1000

    # generate an initial population pool 
    init_pop_pool = np.random.choice([0, -1, 1], size=(sol_per_pop, num_genes))

    prev_1_fitness = 0 
    prev_2_fitness = 0 
    perfect_fitness_bool = False

    for run_cycle in list(range(0, max_iter)):
        fitness_function = min_features_fitness_func
        ga_instance_min = pygad.GA(num_generations=num_generations,
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
                        gene_space = [-1, 0, 1])
        ga_instance_min.run()
        first_solution, first_solution_fitness, first_solution_idx = ga_instance_min.best_solution()
        
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
            gene_space = [-1, 0, 1])
        ga_instance_max.run()
        second_solution, second_solution_fitness, second_solution_idx = ga_instance_max.best_solution()
        
        # If both solutions would output perfect reachability, then pick the one preferred by the user
        if second_solution_fitness >= perfect_fitness and first_solution_fitness >= perfect_fitness:
            if max_edge_first == False: 
                solution = first_solution 
                solution_fitness = first_solution_fitness
                init_pop_pool = ga_instance_min.population
            else:
                solution = second_solution
                solution_fitness = second_solution_fitness
                init_pop_pool = ga_instance_max.population
        else: 
            if second_solution_fitness > first_solution_fitness:  
                solution = second_solution
                solution_fitness = second_solution_fitness
                init_pop_pool = ga_instance_max.population
            else:
                solution = first_solution 
                solution_fitness = first_solution_fitness
                init_pop_pool = ga_instance_min.population
        
        if solution_fitness >= perfect_fitness: 
            perfect_fitness_bool = True

        # if the fitness doesn't change for 3 rounds, then break 
        if solution_fitness == prev_1_fitness and solution_fitness == prev_2_fitness: 
            break 
        else: 
            prev_1_fitness = prev_2_fitness
            prev_2_fitness = solution_fitness

    new_edges_df = pd.DataFrame()
    
    for i in range(0, training_data.shape[0]):
        if solution[i] == 0:
            continue

        x_str = training_data.index[i]

        if solution[i] == -1: 
            reg_type = "-"
        elif solution[i] == 1: 
            reg_type = "+"
        
        # This is where I would select the top regulator for each state 
        if len(regulators_rank) == 0: 
            for regulator in feature_dict[x_str]:
                temp_edge = pd.DataFrame(data = [[regulator, target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
                new_edges_df = pd.concat([new_edges_df, temp_edge])
        else: #TODO make this a little bit more elegant. Maybe instead of using correlation, find other metrics 
            dup_genes = feature_dict[x_str]
            reg_rank = np.array(range(0, len(regulators_rank)))
            intersect_index = np.where(np.isin(regulators_rank, dup_genes))

            intersect_genes = regulators_rank[intersect_index]
            reg_rank = reg_rank[intersect_index]
            intersect_genes = intersect_genes[np.argsort(reg_rank)]

            if len(intersect_genes) > max_dup_genes:
                intersect_genes = intersect_genes[0:max_dup_genes]

            for regulator in intersect_genes:
                temp_edge = pd.DataFrame(data = [[regulator, target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
                new_edges_df = pd.concat([new_edges_df, temp_edge])
            
    return [new_edges_df, perfect_fitness_bool]

def create_network_penalize(training_dict, selected_regulators_dict = dict(), regulators_rank_dict = dict(), unlikley_activators_dict = dict(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, max_edge_first = False, max_dup_genes = 2): 
    total_network = pd.DataFrame()
    for temp_gene in training_dict.keys():
        if temp_gene in selected_regulators_dict.keys():
            selected_regulators_list = selected_regulators_dict[temp_gene]
        else:
            selected_regulators_list = list()

        if temp_gene in regulators_rank_dict.keys():
            regulators_rank_list = regulators_rank_dict[temp_gene]
        else:
            regulators_rank_list = list()

        if temp_gene in unlikley_activators_dict.keys(): 
            unlikely_activators_list = unlikley_activators_dict[temp_gene]
        else:
            unlikely_activators_list = list()
        
        new_network, perfect_fitness_bool = GA_fit_data_penalize(training_dict, 
                                                        temp_gene, 
                                                        selected_regulators = selected_regulators_list, 
                                                        regulators_rank = regulators_rank_list,
                                                        unlikely_activators = unlikely_activators_list,
                                                        num_generations = num_generations, 
                                                        max_iter = max_iter, 
                                                        num_parents_mating = num_parents_mating, 
                                                        sol_per_pop = sol_per_pop, 
                                                        reduce_auto_reg = reduce_auto_reg, 
                                                        max_edge_first = max_edge_first, 
                                                        max_dup_genes = max_dup_genes)
        if perfect_fitness_bool == False:
            new_network, perfect_fitness_bool = GA_fit_data_penalize(training_dict, 
                                            temp_gene, 
                                            regulators_rank = regulators_rank_list,
                                            unlikely_activators = unlikely_activators_list,
                                            num_generations = num_generations, 
                                            max_iter = max_iter, 
                                            num_parents_mating = num_parents_mating, 
                                            sol_per_pop = sol_per_pop, 
                                            reduce_auto_reg = reduce_auto_reg, 
                                            max_edge_first = max_edge_first, 
                                            max_dup_genes = max_dup_genes)
        total_network = pd.concat([total_network, new_network])
        if perfect_fitness_bool == False: 
            print(temp_gene + " does not fit perfectly")
    return total_network
