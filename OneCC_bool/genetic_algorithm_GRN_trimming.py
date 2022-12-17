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

def curate_training_data(state_dict, transition_dict, lineage_time_change_dict, samp_tab, cluster_id = "leiden", pt_id = "dpt_pseudotime",act_tolerance = 0.01, filter_dup_state = True, selected_regulators = list()):
    # this is to prototype the training data required 
    # potential_regulators_dict = dict()
    training_dict = dict()

    # TODO: maybe not hardcode this. There should always have a lineage_0 
    all_genes = transition_dict['lineage_0'].index

    cluster_time_dict = dict()
    for cluster in np.unique(samp_tab[cluster_id]):
        sub_samp_tab = samp_tab.loc[samp_tab[cluster_id] == cluster, :]
        cluster_time_dict[cluster] = np.mean(sub_samp_tab[pt_id])

    #NOTE this is to make the state priority dictionary. I don't think we ever use the state priority for the reconstruction of the network 
    # for the now, let's include those in here 
    state_priority_dict = dict()
    for temp_lineage in transition_dict.keys(): 
        temp_state_df = transition_dict[temp_lineage]
        for i in range(0, temp_state_df.shape[1]): 
            state_priority_dict[temp_state_df.columns[i]] = i + 1
    
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
            prev_index_list = np.where(temp_transition.loc[temp_gene, :] != 0)[0] - 1
            prev_index_list = prev_index_list[prev_index_list > -1]
            
            # find the index of transition of temp_gene 
            cur_index_list = np.where(temp_transition.loc[temp_gene, :] != 0)[0]
            cur_index_list = cur_index_list[cur_index_list > 0]
            
            # all the genes that changed in right before temp_gene transition gets logged 
            # TODO remove anything related to potential_regulators 
            for temp_index in prev_index_list: 
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
            for temp_index in cur_index_list: 
                cell_type_1 = temp_transition.columns[temp_index - 1]
                cell_type_2 = temp_transition.columns[temp_index]
                
                # select the time series data within a range 
                min_time = np.median(samp_tab.loc[samp_tab[cluster_id] == cell_type_1, pt_id]) # TODO could consider median 
                max_time = np.max(samp_tab.loc[samp_tab[cluster_id] == cell_type_2, pt_id]) # TODO probably median might make more sense
                
                # get all the transition  
                gene_transitions = temp_transition.iloc[:, temp_index].copy()
                gene_transitions = gene_transitions[gene_transitions != 0]
                
                sub_temp_time_change = temp_time_change.loc[np.logical_and(temp_time_change['PseudoTime'] >= min_time, temp_time_change['PseudoTime'] <= max_time), ].copy()
 

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
            possible_feature_index_list = np.arange(0, temp_transition.shape[1])
            exclude_indexes = np.setdiff1d(prev_index_list, cur_index_list) # remove the all instance at which the previous configuration occurs 
            
            # loop through the possible feature index list 
            possible_feature_index_list = np.setdiff1d(possible_feature_index_list, exclude_indexes) 

            for possible_feature_index in possible_feature_index_list:
                col_name = temp_state.columns[possible_feature_index]
                cur_col = temp_state.iloc[:, possible_feature_index].copy()

                # if it is indicated that we might have to change the genes
                # find the genes that occured after the change of target gene 
                # use the previous gene expression as feature 
                # TODO change the not_regulator_genes_dict to something like -- occured after target gene change 
                if possible_feature_index in not_regulator_genes_dict.keys():
                    prev_col = temp_state.iloc[:, possible_feature_index - 1].copy()
                    for not_regulator_gene in not_regulator_genes_dict[possible_feature_index]:
                        cur_col[not_regulator_gene] = prev_col[not_regulator_gene]
                
                # this is to check the self regulation features 
                # this is to add in self-regulation 
                # NOTE: this part of the code is to add in the self-regulatory features 
                # TODO: this code is quite confusing. Please make it more readable 
                if possible_feature_index - 1 < 0: 
                    self_regulation_features[col_name] = cur_col[temp_gene]
                else: 
                    prev_col = temp_state.iloc[:, possible_feature_index - 1].copy()
                    if prev_col[temp_gene] == 1: 
                        self_regulation_features[col_name] = 1

                    # if we want to turn off the constant self-activation 
                    else:
                        '''
                        if temp_state.columns[possible_feature_index - 1] in self_regulation_features.keys():
                            if self_regulation_features[temp_state.columns[possible_feature_index - 1]] == 1: 
                                self_regulation_features[col_name] = 1
                            else:
                                self_regulation_features[col_name] = 0
                        else:
                            self_regulation_features[col_name] = 0   
                        '''
                        self_regulation_features[col_name] = 0      
                raw_feature_matrix[col_name] = cur_col
        
            # add the steady state profile as a seperate training data point 
            # to ensure that the steady state will be maintained as steady state
            raw_feature_matrix[temp_state.columns[-1] + "_SteadyState"] = temp_state[temp_state.columns[-1]]
            bad_genes_dict[temp_lineage] = not_regulator_genes_dict    

            # TODO this seems like the most logical place to add the steady state 
            # TODO add logic for adding in the steady states 
        if len(potential_regulators) == 0:
            continue
        training_set = dict()
        training_set['phenotype'] = np.array(raw_feature_matrix.loc[temp_gene, :])
        training_set['bad_genes'] = bad_genes_dict

        # add in the self regulation 
        processed_feature_matrix = raw_feature_matrix.copy()
        for temp_col in processed_feature_matrix.columns:
            if "_SteadyState" in temp_col: 
                processed_feature_matrix.loc[temp_gene, temp_col] = self_regulation_features[temp_col.replace("_SteadyState", "")]
            else:
                processed_feature_matrix.loc[temp_gene, temp_col] = self_regulation_features[temp_col]

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
                
                # if the exact same gene pattern has been used before 
                if temp_feat_pattern in feature_pattern_dict.keys():
                    old_column_name = feature_pattern_dict[temp_feat_pattern]
                    if cluster_time_dict[old_column_name.replace("_SteadyState", "")] < cluster_time_dict[column_name.replace("_SteadyState", "")]:
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
        
        if len(selected_regulators) > 0:
            selected_regulators = list(selected_regulators)
            selected_regulators.append(temp_gene) # add self-regulation 
            if set(selected_regulators).issubset(set(processed_feature_matrix.index)) == True:
                processed_feature_matrix = processed_feature_matrix.loc[processed_feature_matrix.index.isin(selected_regulators), :]            
            else:
                print("Some regulators are not in the expression matrix. All genes are used as regulators")

        training_set['features'] = processed_feature_matrix
        # set up priority list for each state in the case that we would need to assign priority in the genetic algorithm later on 
        state_priority = list()
        for temp_state in training_set['features'].columns:
            if "_SteadyState" in temp_state: 
                temp_state = temp_state.replace("_SteadyState", "")
            state_priority.append(state_priority_dict[temp_state])
        training_set['state_priority'] = np.array(state_priority)
        training_dict[temp_gene] = training_set
    return training_dict

def GA_fit_data_old(training_dict, target_gene, initial_state, selected_regulators = list(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, remove_bad_genes = False, max_edge_first = False): 
    def get_bad_genes(train_dict):
        bad_genes = list()
        first_run = True
        for lineage in train_dict['bad_genes'].keys():
            for cluster in train_dict['bad_genes'][lineage].keys():
                temp_bad_genes = train_dict['bad_genes'][lineage][cluster]
                if first_run == True:
                    bad_genes = list(temp_bad_genes)
                    first_run = False
                else:
                    bad_genes = np.intersect1d(bad_genes, list(temp_bad_genes))
        return bad_genes

    training_data = training_dict[target_gene]['features']
    if len(selected_regulators) > 0: 
        selected_regulators = np.append(selected_regulators, target_gene)
        selected_regulators = np.unique(selected_regulators)
        training_data = training_data.loc[selected_regulators, :]

    if remove_bad_genes == True: 
        all_bad_genes = get_bad_genes(training_dict[target_gene])
        training_data = training_data.loc[~training_data.index.isin(all_bad_genes), :]

    training_data_original = training_data.copy()

    training_data = training_data.loc[training_data.sum(axis = 1) > 0, :] # remove genes that are not active in any states 
    training_data = training_data.drop_duplicates() # remove duplicated genes with the same state profiles 
    
    training_targets = training_dict[target_gene]['phenotype']
    feature_dict = dict()

    # load in the priority 
    state_priorities = training_dict[target_gene]['state_priority']
    state_weights = np.max(state_priorities) + 1 - state_priorities

    def to_string_pattern(gene_pattern):
        gene_pattern = [str(x) for x in gene_pattern]
        return "_".join(gene_pattern)

    # the below is to 
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

            # give less weight to the initial state -- since initial states could have mutual inhibition that contradict the overall trend
            # allow some room for the initial state to be contradictory. 10 initial match == 1 other match 
            if state == initial_state: 
                temp_score = int(total_prob == training_targets[i])
                temp_score = temp_score * 500
            else:
                temp_score = int(total_prob == training_targets[i]) * 1000

            correctness_sum = correctness_sum + temp_score
        fitness_score = correctness_sum + (np.sum(solution == 1) * 1) + (np.sum(solution == -1) * 1.5) # if a gene can be either activator or inhibitor, choose inhibitor

        # remove self inhibition since it would not work unless we go on to protein level 
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * 1000)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score + 4 # remove unnecessary auto-activator. 
        
        # if it comes to non-reactive and reactive, pick the reactive gene 
        fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) != 0, :].sum()) * 0.01
        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - (3 * 1000)
        return fitness_score

    def min_features_fitness_func(solution, solution_idx):
        correctness_sum = 0
        
        correct_scale = 1000
        activated_bonus_scale = 0.01
        
        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            if state == initial_state: 
                temp_score = int(total_prob == training_targets[i])
                temp_score = temp_score * (correct_scale/2)
            else:
                temp_score = int(total_prob == training_targets[i]) * correct_scale
            correctness_sum = correctness_sum + temp_score

        #fitness_score = correctness_sum + (np.sum(solution == 0) * 1) + (np.sum(solution == -1) * 0.1) #if an edge can be either activator or inhibitor, choose activator 
        fitness_score = correctness_sum + (np.sum(solution == 0) * 1) #minimize the number of edges 

        # penalize the self inhibitors
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * correct_scale)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score + 4 # remove unnecessary auto-activator. 

        fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) != 0, :].sum()) * activated_bonus_scale # favor genes that are turned on across more states 
        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - 3000
        return fitness_score

    # the below are just parameters for the genetic algorithm  

    num_genes = training_data.shape[0]

    parent_selection_type = "sss"
    keep_parents = num_parents_mating
    crossover_type = "uniform"
    
    mutation_type = "random"
    mutation_percent_genes = 40
    #mutation_percent_genes = [40, 10] # maybe try to use number of genes as 

    if initial_state in training_data.columns: 
        perfect_fitness = (training_data.shape[1] - 1) * 1000 + 500
    else: 
        perfect_fitness = training_data.shape[1] * 1000

    # TODO generate an initial population pool 
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
        
        # for prototyping purposes 
        if second_solution_fitness >= perfect_fitness and first_solution_fitness >= perfect_fitness:
            #TODO 
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
            
        '''
        elif solution_fitness >= perfect_fitness: 
            break
        '''
        if solution_fitness == prev_1_fitness and solution_fitness == prev_2_fitness: 
            break 
        else: 
            prev_1_fitness = prev_2_fitness
            prev_2_fitness = solution_fitness

    new_edges_df = pd.DataFrame()
    
    if perfect_fitness_bool == False: 
        print(target_gene + " does not fit perfectly")
        print(str(solution_fitness) + "/" + str(perfect_fitness))
    
    for i in range(0, training_data.shape[0]):
        if solution[i] == 0:
            continue

        x_str = training_data.index[i]

        if solution[i] == -1: 
            reg_type = "-"
        elif solution[i] == 1: 
            reg_type = "+"
            
        for regulator in feature_dict[x_str]:
            temp_edge = pd.DataFrame(data = [[regulator, target_gene, reg_type]], columns = ['TF', 'TG', "Type"])
            new_edges_df = pd.concat([new_edges_df, temp_edge])
            
    return [new_edges_df, perfect_fitness_bool]

# have the parameters in 
def create_network_old(training_dict, initial_state, selected_regulators_dict = dict(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, remove_bad_genes = False, max_edge_first = False): 
    total_network = pd.DataFrame()
    for temp_gene in training_dict.keys():
        print("start fitting " + temp_gene )
        if temp_gene in selected_regulators_dict.keys():
            new_network, perfect_fitness_bool = GA_fit_data_old(training_dict, 
                                                            temp_gene, 
                                                            initial_state, 
                                                            selected_regulators = selected_regulators_dict[temp_gene], 
                                                            num_generations = num_generations, 
                                                            max_iter = max_iter, 
                                                            num_parents_mating = num_parents_mating, 
                                                            sol_per_pop = sol_per_pop, 
                                                            reduce_auto_reg = reduce_auto_reg, 
                                                            remove_bad_genes = remove_bad_genes, 
                                                            max_edge_first = max_edge_first)
            if perfect_fitness_bool == False: # if the fitness does not satisify the reachability of all states, then use all the genes 
                new_network, perfect_fitness_bool = GA_fit_data_old(training_dict, 
                                                                temp_gene, 
                                                                initial_state, 
                                                                selected_regulators = list(), 
                                                                num_generations = num_generations, 
                                                                max_iter = max_iter, 
                                                                num_parents_mating = num_parents_mating, 
                                                                sol_per_pop = sol_per_pop, 
                                                                reduce_auto_reg = reduce_auto_reg, 
                                                                remove_bad_genes = remove_bad_genes, 
                                                                max_edge_first = max_edge_first)
        else:
            new_network, perfect_fitness_bool = GA_fit_data_old(training_dict, 
                                                            temp_gene, 
                                                            initial_state, 
                                                            selected_regulators = list(), 
                                                            num_generations = num_generations, 
                                                            max_iter = max_iter, 
                                                            num_parents_mating = num_parents_mating, 
                                                            sol_per_pop = sol_per_pop, 
                                                            reduce_auto_reg = reduce_auto_reg, 
                                                            remove_bad_genes = remove_bad_genes, 
                                                            max_edge_first = max_edge_first)
        total_network = pd.concat([total_network, new_network])
    return total_network


def GA_fit_data(training_dict, target_gene, initial_state, selected_regulators = list(), regulators_rank = list(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, remove_bad_genes = False, max_edge_first = False, max_dup_genes = 2): 
    def get_bad_genes(train_dict):
        bad_genes = list()
        first_run = True
        for lineage in train_dict['bad_genes'].keys():
            for cluster in train_dict['bad_genes'][lineage].keys():
                temp_bad_genes = train_dict['bad_genes'][lineage][cluster]
                if first_run == True:
                    bad_genes = list(temp_bad_genes)
                    first_run = False
                else:
                    bad_genes = np.intersect1d(bad_genes, list(temp_bad_genes))
        return bad_genes

    training_data = training_dict[target_gene]['features']
    if len(selected_regulators) > 0: 
        selected_regulators = np.append(selected_regulators, target_gene)
        selected_regulators = np.unique(selected_regulators)
        training_data = training_data.loc[selected_regulators, :]

    if remove_bad_genes == True: 
        all_bad_genes = get_bad_genes(training_dict[target_gene])
        training_data = training_data.loc[~training_data.index.isin(all_bad_genes), :]

    training_data_original = training_data.copy()

    training_data = training_data.loc[training_data.sum(axis = 1) > 0, :] # remove genes that are not active in any states 
    training_data = training_data.drop_duplicates() # remove duplicated genes with the same state profiles 
    
    training_targets = training_dict[target_gene]['phenotype']
    feature_dict = dict()

    # load in the priority 
    state_priorities = training_dict[target_gene]['state_priority']
    state_weights = np.max(state_priorities) + 1 - state_priorities

    def to_string_pattern(gene_pattern):
        gene_pattern = [str(x) for x in gene_pattern]
        return "_".join(gene_pattern)

    # the below is to 
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

            # give less weight to the initial state -- since initial states could have mutual inhibition that contradict the overall trend
            # allow some room for the initial state to be contradictory. 10 initial match == 1 other match 
            if state == initial_state: 
                temp_score = int(total_prob == training_targets[i])
                temp_score = temp_score * 500
            else:
                temp_score = int(total_prob == training_targets[i]) * 1000

            correctness_sum = correctness_sum + temp_score
        fitness_score = correctness_sum + (np.sum(solution == 1) * 1) + (np.sum(solution == -1) * 1.5) # if a gene can be either activator or inhibitor, choose inhibitor

        # remove self inhibition since it would not work unless we go on to protein level 
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * 1000)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score + 4 # remove unnecessary auto-activator. 
                else:
                    fitness_score = fitness_score - 4
        # if it comes to non-reactive and reactive, pick the reactive gene 
        fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) != 0, :].sum()) * 0.01
        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - (3 * 1000)
        return fitness_score

    def min_features_fitness_func(solution, solution_idx):
        correctness_sum = 0
        
        correct_scale = 1000
        activated_bonus_scale = 0.01
        
        for i in range(0, len(training_data.columns)):
            state = training_data.columns[i]
            norm_dict = training_data.loc[:, state]
            upTFs = training_data.index[np.array(solution) == 1]
            downTFs = training_data.index[np.array(solution) == -1]
            activation_prob = calc_activation_prob(norm_dict.to_dict(), upTFs)
            repression_prob = calc_repression_prob(norm_dict.to_dict(), downTFs)

            total_prob = activation_prob * repression_prob

            if state == initial_state: 
                temp_score = int(total_prob == training_targets[i])
                temp_score = temp_score * (correct_scale/2)
            else:
                temp_score = int(total_prob == training_targets[i]) * correct_scale
            correctness_sum = correctness_sum + temp_score

        #fitness_score = correctness_sum + (np.sum(solution == 0) * 1) + (np.sum(solution == -1) * 0.1) #if an edge can be either activator or inhibitor, choose activator 
        fitness_score = correctness_sum + (np.sum(solution == 0) * 1) #minimize the number of edges 

        # penalize the self inhibitors
        if self_reg_index > -1:
            if solution[self_reg_index] == -1:
                fitness_score = fitness_score - (3 * correct_scale)
            elif solution[self_reg_index] == 1:
                if reduce_auto_reg == False:
                    fitness_score = fitness_score + 4 # remove unnecessary auto-activator. 
                else:
                    fitness_score = fitness_score - 4
        fitness_score = fitness_score + np.sum(training_data.loc[np.array(solution) != 0, :].sum()) * activated_bonus_scale # favor genes that are turned on across more states 
        if np.sum(np.abs(solution)) == 0: # if there are no regulation on the target gene, not even self regulation, then it's not acceptable
            fitness_score = fitness_score - 3000
        return fitness_score

    # the below are just parameters for the genetic algorithm  

    num_genes = training_data.shape[0]

    parent_selection_type = "sss"
    keep_parents = num_parents_mating
    crossover_type = "uniform"
    
    mutation_type = "random"
    mutation_percent_genes = 40
    #mutation_percent_genes = [40, 10] # maybe try to use number of genes as 

    if initial_state in training_data.columns: 
        perfect_fitness = (training_data.shape[1] - 1) * 1000 + 500
    else: 
        perfect_fitness = training_data.shape[1] * 1000

    # TODO generate an initial population pool 
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
        
        # for prototyping purposes 
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
            
        '''
        elif solution_fitness >= perfect_fitness: 
            break
        '''
        if solution_fitness == prev_1_fitness and solution_fitness == prev_2_fitness: 
            break 
        else: 
            prev_1_fitness = prev_2_fitness
            prev_2_fitness = solution_fitness

    new_edges_df = pd.DataFrame()
    
    if perfect_fitness_bool == False: 
        print(target_gene + " does not fit perfectly")
        print(str(solution_fitness) + "/" + str(perfect_fitness))
    
    for i in range(0, training_data.shape[0]):
        if solution[i] == 0:
            continue

        x_str = training_data.index[i]

        if solution[i] == -1: 
            reg_type = "-"
        elif solution[i] == 1: 
            reg_type = "+"
        
        #TODO This is where I would select the top regulator for each state 
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

# have the parameters in 
def create_network(training_dict, initial_state, selected_regulators_dict = dict(), regulators_rank_dict = dict(), num_generations = 1000, max_iter = 10, num_parents_mating = 4, sol_per_pop = 10, reduce_auto_reg = True, remove_bad_genes = False, max_edge_first = False, max_dup_genes = 2): 
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
                                                        initial_state, 
                                                        selected_regulators = selected_regulators_list, 
                                                        regulators_rank = regulators_rank_list,
                                                        num_generations = num_generations, 
                                                        max_iter = max_iter, 
                                                        num_parents_mating = num_parents_mating, 
                                                        sol_per_pop = sol_per_pop, 
                                                        reduce_auto_reg = reduce_auto_reg, 
                                                        remove_bad_genes = remove_bad_genes, 
                                                        max_edge_first = max_edge_first, 
                                                        max_dup_genes = max_dup_genes)
        if perfect_fitness_bool == False:
            new_network, perfect_fitness_bool = GA_fit_data(training_dict, 
                                            temp_gene, 
                                            initial_state, 
                                            regulators_rank = regulators_rank_list,
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
