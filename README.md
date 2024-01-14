# OneSC (One Synthetic Cell) 

### <a name="introduction">Introduction</a>
OneSC is an computational tool for inferring and simulating core transcription factors circuits. 

Below is a walk-through tutorial on 
1. how to infer the transcription factors circuit
2. how to simulate synthetic single cells across developmental trajectories using the circuit 

### Table of contents

[Installation](#installation) <br>

[Inference of GRN](#grn_inference) <br>

[Simulation of Synthetic Cells](#simulateSynCells)

[Perform Perturbation Simulation](#perturbSynCells)

[Optional - Identification of dynamic TFs](#identifyDynTFs)

### <a name="installation">Installation</a>

### <a name="grn_inference">Inference of GRN</a>
In the tutorial, we are going to use the mouse myeloid single-cell data from [Paul et al, 2015](https://www.cell.com/cell/fulltext/S0092-8674(15)01493-2?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867415014932%3Fshowall%3Dtrue). You can download the [expression profiles of core transcription factors]() and the [sample table]() with pusedotime and cluster information. 

Import the required packages. 
```
import numpy as np 
import pandas as pd 
import onesc 
import networkx as nx
```

Load in the training data. 
```
train_exp = pd.read_csv("train_exp.csv", index_col = 0)
samp_tab = pd.read_csv("samp_tab.csv", index_col = 0)
pt_col = 'dpt_pseudotime'
cluster_col = 'cell_types'
```

Construct the cluster-cluster transition graph. 
```
initial_clusters = ['CMP']
end_clusters = ['Erythrocytes', 'Granulocytes', 'Monocytes', 'MK']
clusters_G = onesc.construct_cluster_network(train_exp, samp_tab, initial_clusters = initial_clusters, terminal_clusters = end_clusters, cluster_col = cluster_col, pseudo_col = pt_col)
```

We can visualize the networkx strcture of cluster-cluster transition graph. 
```
nx.draw(clusters_G, with_labels = True)
```
<img src="img/cluster_cluster_graph.png">

Run the GRN inference step and save the GRN. 
```
# extract inidividual trajectories found in the data 
lineage_cluster = onesc.extract_trajectory(clusters_G,initial_clusters, end_clusters)

# find the boolean threshold for each gene 
vector_thresh = onesc.find_threshold_vector(train_exp, samp_tab, cluster_col = "cell_types", cutoff_percentage=0.4)

# identify the finner time steps at which genes change along individual trajectory 
lineage_time_change_dict = onesc.find_gene_change_trajectory(train_exp, samp_tab, lineage_cluster, cluster_col, pt_col, vector_thresh, pseudoTime_bin=0.01) 

# define boolean states profiles for each cell cluster 
state_dict = onesc.define_states(train_exp, samp_tab, lineage_cluster, vector_thresh, cluster_col, percent_exp = 0.3)

# define transition profiles for each cell clusters
transition_dict = onesc.define_transition(state_dict)

# curate the training data for inference of GRN for each gene 
training_data = onesc.curate_training_data(state_dict, transition_dict, lineage_time_change_dict, samp_tab, cluster_id = cluster_col, pt_id = pt_col,act_tolerance = 0.04)

# calculate the pearson correlation between genes. This adds more information during the inference step. 
corr_mat = onesc.calc_corr(train_exp)

# infer the gene regulatory network
ideal_edge_num = round(0.4 * corr_mat.shape[1])
inferred_grn = onesc.create_network(training_data, 
                                    corr_mat, 
                                    ideal_edges = ideal_edge_num, 
                                    num_generations = 300, 
                                    max_iter = 30, 
                                    num_parents_mating = 4, 
                                    sol_per_pop = 30, 
                                    reduce_auto_reg = True)
inferred_grn.to_csv("OneSC_network.csv")
```
You can print the inferred GRN out. 
```
print(inferred_grn)

#       TF     TG Type
#0    Fli1  Cebpa    -
#1   Gata1  Cebpa    -
#2   Gfi1b  Cebpa    -
#3    Klf1  Cebpa    -
#4   Zfpm1  Cebpa    -
#5   Gata1  Cebpe    -
#6   Gata2  Cebpe    -
#7    Irf8  Cebpe    -
# ...
```
