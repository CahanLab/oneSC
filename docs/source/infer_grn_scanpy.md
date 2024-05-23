# Inference of functional transcription factors network (AnnData Object)
In this tutorial, we are going to show how to run OneSC using AnnData object. We are going to use the mouse myeloid single-cell data from [Paul et al, 2015](https://pubmed.ncbi.nlm.nih.gov/26627738/). We have refined the annotation of these 2,670 cells. Please download the [h5ad file containing the expression values of 12 core transcription factors in these cells here](https://cnobjects.s3.amazonaws.com/OneSC/0.1.0/Paul15_040824.h5ad). Note that this annData object includes cell type annotation and precomputed pseudotime metadata in adata.obs['cell_types'] and adata.obs['dpt_pseudotime'], respectively.  

### Setup
Launch Jupyter or your Python interpreter. Import the required packages and functions.
```
import numpy as np 
import pandas as pd 
import onesc 
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scanpy as sc
import anndata
import scipy as sp
import pySingleCellNet as pySCN
from joblib import dump, load
import sys
import igraph as ig
from igraph import Graph
ig.config['plotting.backend'] = 'matplotlib'
```

Load in the training data:
```
adata = sc.read_h5ad("Paul15_040824.h5ad")
```
### GRN inference 
The first step in reconstructing or inferring a GRN with oneSC is to determine the directed state graph of the cells. In other words, what is the sequence of distinct states that a cell passes through from the start to a terminal state? oneSC requires that the user provide cell state annotations. Typically these are in the form of cell clusters or cell type annotations. oneSC also requires that the user specify the initial cell states and the end states. In our data, the cell states have already been provided in  .obs['cell_types']. Now, we will specify the initial cell states and the end states:
```
initial_clusters = ['CMP']
end_clusters = ['Erythrocytes', 'Granulocytes', 'Monocytes', 'MK']
```

We can use oneSC to infer the directed state graph since it knows the initial and terminal states and the pseudotime of all cells:
```
state_path = onesc.construct_cluster_graph_adata(adata, initial_clusters = initial_clusters, terminal_clusters = end_clusters, cluster_col = "cell_types", pseudo_col = "dpt_pseudotime")

onesc.plot_state_graph(state_path)
```

![State graph](./_static/images/state_graph_1.png)

However, you can also manually create a directed state graph:
```
edge_list = [("CMP", "MK"), ("CMP", "MEP"), ("MEP", "Erythrocytes"), ("CMP", "GMP"), ("GMP", "Granulocytes"), ("GMP", "Monocytes")]
H = nx.DiGraph(edge_list)
onesc.plot_state_graph(H)
```

Now we are ready to infer the GRN. There are quite a few parameters to `infer_grn()`. Listed below are required parameters, and those that you can adjust to optimize runtime on your platform. In the example below, we have selected parameter values appropriate for this data.

- cellstate_graph: this is just the state graph we made earlier, H.
- start_end_clusters: a dict of 'start', and 'end' cell states.
- adata: the training data.
- run_parallel: (bool, optional): whether to run network inference in parallel. Defaults to True
- n_cores (int, optional): number of cores to run the network inference in parallel. Defaults to 16

infer_grn() returns a Pandas DataFrame. We convert it to an igraph graph for visualization.

```
start_end_states = {'start': ['CMP'], 'end':['MK', 'Erythrocytes', 'Granulocytes', 'Monocytes']}

iGRN = onesc.infer_grn(H, start_end_states, adata, num_generations = 20, sol_per_pop = 30, reduce_auto_reg=True, ideal_edges = 0, GA_seed_list = [1, 3], init_pop_seed_list = [21, 25], cluster_col='cell_types', pseudoTime_col='dpt_pseudotime')

grn_ig = onesc.dataframe_to_igraph(iGRN)
onesc.plot_grn(grn_ig, layout_method='fr',community_first=True)
```
![CMP GRN](./_static/images/CMP_grn.png)

The purple edges represent positive regulatory relationships (i.e. TF promotes expression of TG), whereas grey edges represent inhibitory relationships. Nodes have been colored by a community detection algorithm applied to the GRN.
