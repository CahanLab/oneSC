from concurrent.futures import process
import numpy as np 
import pandas as pd
import pygad

def define_states(exp_tab, samp_tab, lineage_cluster, vector_thresh, cluster_col = 'cluster_id'):
    state_dict = dict()
    for lineage in lineage_cluster.keys():
        
        temp_df = pd.DataFrame()
        for cell_type in lineage_cluster[lineage]:
            sub_st = samp_tab.loc[samp_tab[cluster_col] == cell_type, :]
            sub_exp = exp_tab.loc[:, sub_st.index]
            temp_df[cell_type] = (sub_exp.mean(axis = 1) > vector_thresh) * 1
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

def curate_training_data(state_dict, transition_dict, lineage_time_change_dict, samp_tab, cluster_id = "leiden", pt_id = "dpt_pseudotime",act_tolerance = 0.01, filter_dup_state = True):
    # this is to prototype the training data required 
    # potential_regulators_dict = dict()
    training_dict = dict()

    # TODO: maybe not hardcode this
    all_genes = transition_dict['lineage_0'].index

    cluster_time_dict = dict()
    for cluster in np.unique(samp_tab[cluster_id]):
        sub_samp_tab = samp_tab.loc[samp_tab[cluster_id] == cluster, :]
        cluster_time_dict[cluster] = np.mean(sub_samp_tab[pt_id])

    for temp_gene in all_genes: 

        # get all the potential regulators 
        potential_regulators = list() # TODO remove the potential regulators list and remove the prev index 
        raw_feature_matrix = pd.DataFrame()
        self_regulation_features = dict()

        # this is to store the bad genes 
        bad_genes_dict = dict()

        # loop through all the lineages to find appropriate potential regulators 
        for temp_lineage in transition_dict.keys():

            temp_transition = transition_dict[temp_lineage] # transition matrix 
            temp_state = state_dict[temp_lineage] # state matrix 
            temp_time_change = lineage_time_change_dict[temp_lineage] # time series 
            
            # find the index of the state before transition of temp_gene 
            prev_index = np.where(temp_transition.loc[temp_gene, :] != 0)[0] - 1
            prev_index = prev_index[prev_index > -1]
            
            # find the index of transition of temp_gene 
            cur_index = np.where(temp_transition.loc[temp_gene, :] != 0)[0]
            cur_index = cur_index[cur_index > 0]
            
            # all the genes that changed in right before temp_gene transition gets logged 
            for temp_index in prev_index: 
                gene_transitions = temp_transition.iloc[:, temp_index].copy()
                gene_transitions_2 = temp_transition.iloc[:, temp_index + 1].copy()
                
                gene_transitions = gene_transitions[gene_transitions != 0]
                gene_transitions_2 = gene_transitions_2[gene_transitions_2 != 0] # check if the gene immediately changes at the current state 
                
                # select the genes that ONLY change in the previous state.
                # with regards to the genes that also in the current state, there will be more strigent checks on that 
                good_genes = np.setdiff1d(gene_transitions.index, gene_transitions_2.index)
                potential_regulators = np.concatenate([potential_regulators, good_genes])
            
            # designate non-target genes 
            not_regulator_genes_dict = dict()

            # select the appropriate genes that change with the temp_gene 
            for temp_index in cur_index: 
                cell_type_1 = temp_transition.columns[temp_index - 1]
                cell_type_2 = temp_transition.columns[temp_index]
                
                # select the time series data within a range 
                min_time = np.mean(samp_tab.loc[samp_tab[cluster_id] == cell_type_1, pt_id]) # TODO could consider median 
                max_time = np.max(samp_tab.loc[samp_tab[cluster_id] == cell_type_2, pt_id]) # TODO probably median might make more sense
                
                # get all the transition  
                gene_transitions = temp_transition.iloc[:, temp_index].copy()
                gene_transitions = gene_transitions[gene_transitions != 0]
                
                sub_temp_time_change = temp_time_change.loc[np.logical_and(temp_time_change['PseudoTime'] >= min_time, temp_time_change['PseudoTime'] <= max_time), ]
                
                # remove all the disagreement of transition 
                # if temp_gene is suppose to transition into 1, then we remove all the instance at which temp_gene is transitioned into - from the time series 
                if gene_transitions[temp_gene] == 1: 
                    sub_temp_time_change = sub_temp_time_change.loc[np.logical_and(sub_temp_time_change['gene'] == temp_gene, sub_temp_time_change['type'] == "-") == False, :]
                else:
                    sub_temp_time_change = sub_temp_time_change.loc[np.logical_and(sub_temp_time_change['gene'] == temp_gene, sub_temp_time_change['type'] == "+") == False, :]
                
                # sort the values, and assign index 
                sub_temp_time_change = sub_temp_time_change.sort_values("PseudoTime")
                sub_temp_time_change.index = np.arange(0, sub_temp_time_change.shape[0])

                # if the temp_gene is in the data 
                # check if the sign matches 

                if np.sum(sub_temp_time_change['gene'].isin([temp_gene])) > 0: 
                    if gene_transitions[temp_gene] == 1: 
                        sign = "+"
                    else: 
                        sign = "-"
                    # if there are multiple 
                    if np.sum(np.logical_and(sub_temp_time_change['gene'] == temp_gene, sub_temp_time_change['type'] == sign)) != 0:
                        last_index = sub_temp_time_change['gene'].where(np.logical_and(sub_temp_time_change['gene'] == temp_gene, sub_temp_time_change['type'] == sign)).last_valid_index()
                    else:
                        # if the gene did not have the right change within the time frame 
                        # skip...
                        not_regulator_genes = np.array(gene_transitions.index)
                        not_regulator_genes_dict[temp_index] = not_regulator_genes[not_regulator_genes != temp_gene]
                        continue
                else: 
                    # if the target gene didn't even change within the time frame..aka before all the other genes 
                    # skip this whole thing 
                    not_regulator_genes = np.array(gene_transitions.index)
                    not_regulator_genes_dict[temp_index] = not_regulator_genes[not_regulator_genes != temp_gene]
                    continue

                cur_latest_time = sub_temp_time_change.loc[:, "PseudoTime"]
                cur_latest_time = cur_latest_time[last_index] # find all the genes that changed before then
        
                cur_potential_regulators = np.array([])
                for temp_gene_2 in gene_transitions.index:
                    if temp_gene_2 == temp_gene: 
                        continue 
                        
                    if gene_transitions[temp_gene_2] == 1: 
                        sign = "+"
                    else: 
                        sign = "-"
                    
                    # if there exist a time point which the supposed change matches the time-series data 
                    # which means the change occured within an observable time frame 
                    if np.sum(np.logical_and(sub_temp_time_change['gene'] == temp_gene_2, sub_temp_time_change['type'] == sign)) > 0: 
                        sub_temp_time_change_2 = sub_temp_time_change.loc[np.logical_and(sub_temp_time_change['gene'] == temp_gene_2, sub_temp_time_change['type'] == sign), :]
                        sub_temp_time_change_2 = sub_temp_time_change_2.sort_values("PseudoTime") 
                        sub_temp_time_change_2.index = np.arange(0, sub_temp_time_change_2.shape[0])

                        latest_time = sub_temp_time_change_2.loc[:, "PseudoTime"] 
                        latest_time = latest_time[0]
                        # if the change time of potential regulator is much earlier than the change time of temp_gene
                        if latest_time <= (cur_latest_time - act_tolerance):
                            cur_potential_regulators = np.concatenate([cur_potential_regulators, [temp_gene_2]])   
                        else:
                            continue                              
                    else: 
                        # if there is not time point match, that suggest the change probably occured earlier than the observable time frame 
                        latest_time = sub_temp_time_change.loc[:, "PseudoTime"] 
                        latest_time = latest_time[latest_time.index[0]]
                        if latest_time <= (cur_latest_time - act_tolerance):
                            cur_potential_regulators = np.concatenate([cur_potential_regulators, [temp_gene_2]])    

                not_regulator_genes = np.setdiff1d(gene_transitions.index, cur_potential_regulators)  
                not_regulator_genes = np.array(not_regulator_genes)
                not_regulator_genes_dict[temp_index] = not_regulator_genes[not_regulator_genes != temp_gene]

            potential_regulators = np.unique(potential_regulators)

            # this is to extract the features for training logicistic regressor 
            possible_feature_index = np.arange(0, temp_transition.shape[1])
            exclude_indexes = np.setdiff1d(prev_index, cur_index) 
            
            # loop through the possible feature index list 
            possible_feature_index_list = np.setdiff1d(possible_feature_index, exclude_indexes) 
            
            for possible_feature_index in possible_feature_index_list:
                col_name = temp_state.columns[possible_feature_index]
                cur_col = temp_state.iloc[:, possible_feature_index].copy()

                # if it is indicated that we might have to change the genes
                if possible_feature_index in not_regulator_genes_dict.keys():
                    prev_col = temp_state.iloc[:, possible_feature_index - 1].copy()
                    for not_regulator_gene in not_regulator_genes_dict[possible_feature_index]:
                        cur_col[not_regulator_gene] = prev_col[not_regulator_gene]
                
                # this is to check the self regulation features 
                if possible_feature_index - 1 < 0: 
                    self_regulation_features[col_name] = cur_col[temp_gene]
                else: 
                    prev_col = temp_state.iloc[:, possible_feature_index - 1].copy()
                    if prev_col[temp_gene] == 1: 
                        self_regulation_features[col_name] = 1
                    else: 
                        self_regulation_features[col_name] = 0

                raw_feature_matrix[col_name] = cur_col

            bad_genes_dict[temp_lineage] = not_regulator_genes_dict    
        
        # check if the gene is expressed in the very beginning 
        # TODO maybe make it not not be hardcoded by 'lineage_0'
        init_genes = transition_dict['lineage_0'].iloc[:, 0]
        init_genes = init_genes[init_genes == 1].index
        if temp_gene in init_genes: 
            init_state_name = transition_dict['lineage_0'].columns[0]
            init_state = transition_dict['lineage_0'].loc[:, init_state_name].replace(-1, 0)
            raw_feature_matrix[init_state_name] = init_state
            self_regulation_features[init_state_name] = init_state[temp_gene]

        if len(potential_regulators) == 0:
            continue
        training_set = dict()
        training_set['phenotype'] = np.array(raw_feature_matrix.loc[temp_gene, :])
        training_set['bad_genes'] = bad_genes_dict

        # add in the self regulation 
        processed_feature_matrix = raw_feature_matrix.copy()
        processed_feature_matrix.loc[temp_gene, :] = self_regulation_features

        #potential_regulators = potential_regulators[potential_regulators != temp_gene]
        potential_regulators = list(potential_regulators)
        potential_regulators.append(temp_gene)
        potential_regulators = np.unique(potential_regulators)

        #training_set['features'] = processed_feature_matrix.loc[potential_regulators, :]

        # check if the states should be unique 
        if filter_dup_state == True: 
            feature_pattern_dict = dict()
            for column_name in processed_feature_matrix.columns: 
                temp_processed_feature = processed_feature_matrix.loc[:, column_name]
                temp_feat_list = list(temp_processed_feature)
                temp_feat_list = [str(x) for x in temp_feat_list]
                temp_feat_pattern = "-".join(temp_feat_list)
                
                if temp_feat_pattern in feature_pattern_dict.keys():
                    old_colum_name = feature_pattern_dict[temp_feat_pattern]
                    if cluster_time_dict[old_colum_name] < cluster_time_dict[column_name]:
                        feature_pattern_dict[temp_feat_pattern] = column_name
                else:
                    feature_pattern_dict[temp_feat_pattern] = column_name

            # only select the unique states in phenotype 
            # the order of the state should be preserved 
            # unique_state_index = [x in list(feature_pattern_dict.values()) for x in processed_feature_matrix.columns]
            #training_set['phenotype'] = training_set['phenotype'][unique_state_index]
            temp_pheno_list = training_set['phenotype']
            
            pheno_dict = dict()
            for i in range(0, len(temp_pheno_list)):
                temp_state = processed_feature_matrix.columns[i]
                pheno_dict[temp_state] = temp_pheno_list[i]
            
            new_pheno_list = np.array([])
            for temp_state in list(feature_pattern_dict.values()):
                new_pheno_list = np.append(new_pheno_list, pheno_dict[temp_state])

            training_set['phenotype'] = new_pheno_list

            # only select the unique states in feature matrix 
            # the order is not preserved...
            processed_feature_matrix = processed_feature_matrix.loc[:, list(feature_pattern_dict.values())]
            training_set['features'] = processed_feature_matrix

        else: 
            training_set['features'] = processed_feature_matrix

        training_dict[temp_gene] = training_set
    return training_dict

def GA_fit_data(training_dict, target_gene, initial_state, selected_regulators = list(), num_generations = 1000, num_parents_mating = 4, sol_per_pop = 10): 
    # TODO for the future segment the generations out so that you can do them in batches     

    training_data = training_dict[target_gene]['features']
    training_data = training_data.loc[training_data.sum(axis = 1) > 0, :]

    if len(selected_regulators) > 0: 
        selected_regulators = np.append(selected_regulators, target_gene)
        selected_regulators = np.unique(selected_regulators)
        training_data = training_data.loc[selected_regulators, :]

    training_data_original = training_data.copy()
    
    training_data = training_data.drop_duplicates()
    training_targets = training_dict[target_gene]['phenotype']
    feature_dict = dict()

    for regulator in training_data_original.index: 
        gene_pattern = training_data_original.loc[regulator, :]
        gene_pattern = np.array(gene_pattern)
        gene_pattern = [str(x) for x in gene_pattern]
        x_str = "_".join(gene_pattern)
        if x_str in feature_dict.keys():
            feature_dict[x_str] = np.concatenate([feature_dict[x_str], [regulator]])
        else:
            feature_dict[x_str] = np.array([regulator])
            
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

            # give less weight to the initial state -- since initial states could have mutual inhibition that contradict the overall trend
            # allow some room for the initial state to be contradictory. 10 initial match == 1 other match 
            if state == initial_state: 
                temp_score = int(total_prob == training_targets[i])
                temp_score = temp_score * 0.1
            else:
                temp_score = int(total_prob == training_targets[i])
            correctness_sum = correctness_sum + temp_score

        fitness_score = correctness_sum + (np.sum(np.abs(solution)) * 0.001)
        return fitness_score

    def min_features_fitness_func(solution, solution_idx):
        correctness_sum = 0

        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            # give less weight to the initial state -- since initial states could have mutual inhibition that contradict the overall trend
            # allow some room for the initial state to be contradictory. 10 initial match == 1 other match 
            if state == initial_state: 
                temp_score = int(total_prob == training_targets[i])
                temp_score = temp_score * 0.1
            else:
                temp_score = int(total_prob == training_targets[i])
            correctness_sum = correctness_sum + temp_score

        fitness_score = correctness_sum + (np.sum(np.array(solution) == 0) * 0.001)
        return fitness_score

    # the below are just parameters for the genetic algorithm     
    fitness_function = max_features_fitness_func
    num_genes = training_data.shape[0]

    parent_selection_type = "sss"
    keep_parents = 4
    crossover_type = "uniform"
    
    mutation_type = "adaptive"
    #mutation_percent_genes = 20
    mutation_percent_genes = [40, 10] # maybe try to use number of genes as 

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
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
    ga_instance.run()
    max_feat_solution, max_feat_solution_fitness, max_feat_solution_idx = ga_instance.best_solution()
    solution = max_feat_solution

    if initial_state in training_data.columns: 
        perfect_fitness = training_data.shape[1] - 1 + 0.1
    else: 
        perfect_fitness = training_data.shape[1]
    
    if max_feat_solution_fitness < perfect_fitness: 
        print(target_gene + " does not fit perfectly")
        print(str(max_feat_solution_fitness) + "/" + str(perfect_fitness))

        fitness_function = min_features_fitness_func
        ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_function,
                            sol_per_pop=sol_per_pop,
                            num_genes=num_genes,
                            parent_selection_type=parent_selection_type,
                            keep_parents=keep_parents,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            mutation_percent_genes=mutation_percent_genes, 
                            suppress_warnings = True,
                            initial_population = ga_instance.population,
                            gene_space = [-1, 0, 1])

        ga_instance.run()
        min_feat_solution, min_feat_solution_fitness, min_feat_solution_idx = ga_instance.best_solution()

        if min_feat_solution_fitness < perfect_fitness: 
            print(target_gene + " does not fit perfectly")
            print(str(min_feat_solution_fitness) + "/" + str(perfect_fitness))

        if min_feat_solution_fitness > max_feat_solution_fitness:
            solution = min_feat_solution

    new_edges_df = pd.DataFrame()
    
    for i in range(0, training_data.shape[0]):
        if solution[i] == 0:
            continue
        gene_pattern = np.array(training_data.iloc[i, :])
        gene_pattern = [str(x) for x in gene_pattern]
        x_str = "_".join(gene_pattern)

        if solution[i] == -1: 
            reg_type = "-"
        elif solution[i] == 1: 
            reg_type = "+"
            
        for regulator in feature_dict[x_str]:
            temp_edge = pd.DataFrame(data = [[regulator, target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
            new_edges_df = pd.concat([new_edges_df, temp_edge])
            
    return new_edges_df

# have the parameters in 
def create_network(training_dict, initial_state, selected_regulators = list(), num_generations = 1000, num_parents_mating = 4, sol_per_pop = 10): 
    total_network = pd.DataFrame()
    for temp_gene in training_dict.keys():
        new_network = GA_fit_data(training_dict, temp_gene, initial_state, selected_regulators = selected_regulators[temp_gene], num_generations = num_generations, num_parents_mating = num_parents_mating, sol_per_pop = sol_per_pop)
        total_network = pd.concat([total_network, new_network])
    return total_network

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







