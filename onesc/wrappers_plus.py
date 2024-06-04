import numpy as np 
import pandas as pd
import scanpy as sc
import itertools
import networkx as nx
import anndata as ad
import scipy as sp
from .genetic_algorithm_GRN_trimming import define_states
from .genetic_algorithm_GRN_trimming import define_transition
from .genetic_algorithm_GRN_trimming import curate_training_data
from .genetic_algorithm_GRN_trimming import calc_corr 
from .genetic_algorithm_GRN_trimming import create_network_ensemble
from .lineage_compilation import *
import multiprocessing as mp
import warnings
import igraph as ig
from igraph import Graph
import matplotlib
import matplotlib.pyplot as plt
from adjustText import adjust_text
warnings.simplefilter(action='ignore', category=FutureWarning)


def construct_cluster_graph_adata(adata, **kwargs):
    """_summary_

    Args:
        adata (annData): _description_
    
    Kwargs:
        initial_clusters (list): A list of user defined initial cell states or clusters. 
        terminal_clusters (list): A list of user defined terminal cell states or clusters. 
        cluster_col (str, optional): The column name for column in the sample table containing cell states information. Defaults to "cluster_id".
        pseudo_col (str, optional): The column name for column containing pseudotime information. Defaults to "pseudotime".

    Returns:
        networkx.DiGraph: A networkx directed graph object summarizing the transitional relationship between cell states. 
    """
    sampTab = adata.obs.copy()
    train_exp = adata.to_df().T.copy()
    return construct_cluster_network(train_exp, sampTab, **kwargs)

def plot_grn(grn, layout_method="sugiyama", community_first=False):
    """Plot inferred gene regulatory network. 

    Args:
        grn (igraph.Graph): igraph.graph object representing inferred gene regulatory network. 
        layout_method (str, optional): Plotting layout. Defaults to "sugiyama".
        community_first (bool, optional): Whether to perfect community detection when plotting. Defaults to False.
    """
    grn_ig = grn.copy()
    v_style_grn = {}
    v_style_grn["layout"] = grn_ig.layout(layout_method)
    v_style_grn["vertex_size"] = 20
    v_style_grn["edge_width"] = [.5 if edge_type == "-" else 1 for edge_type in grn_ig.es["Type"]]
    v_style_grn["edge_color"] = ["#7142cf" if edge_type == "+" else "#AAA" for edge_type in grn_ig.es["Type"]]
    v_style_grn["edge_arrow_width"] = 5 

    # Handle community detection and coloring
    if community_first:
        neg_edges = grn_ig.es.select(Type='-')
        grn_pos = grn_ig.copy()
        grn_pos.delete_edges(neg_edges)
        communities = grn_pos.community_edge_betweenness()
        communities = communities.as_clustering()

        v_style_grn["layout"] = grn_pos.layout(layout_method)

        num_communities = len(communities)
        palette = ig.RainbowPalette(n=num_communities)

        for i, community in enumerate(communities):
            color = palette.get(i)
            for vertex in community:
                grn_ig.vs[vertex]["color"] = color

    # Plot the graph with matplotlib backend
    fig, ax = plt.subplots()
    ig.plot(grn_ig, **v_style_grn, target=ax)

    # Adjust and place vertex labels
    # Ensure we're only accessing existing vertex indices
    for i in range(len(grn_ig.vs)):
        x, y = v_style_grn["layout"][i]
        label = grn_ig.vs["name"][i] if i < len(grn_ig.vs["name"]) else 'NA'
        plt.text(x, y, label, ha='center', va='center')

    # Use adjustText to optimize text label positions
    # Assuming texts have been defined and added to the plot as shown above
    texts = [text for text in ax.texts]
    adjust_text(texts)

    plt.show()

def plot_state_graph(nx_network, layout = "sugiyama"):
    """Generate plot for state transition graph 

    Args:
        nx_network (networkx.DiGraph): A networkx directed graph object summarizing the transitional relationship between cell states.
        layout (str, optional): The layout style for igraph plotting. Defaults to "sugiyama".
    """
    v_style_trajectory = {}
    v_style_trajectory["layout"] = layout
    v_style_trajectory["vertex_label_dist"] = 2.5
    v_style_trajectory["vertex_label_angle"] = 3 # in radians
    v_style_trajectory["bbox"] = (600,600)
    v_style_trajectory["margin"] = (50)
    ig_net = Graph.from_networkx(nx_network)
    ig_net.vs["label"] = ig_net.vs["_nx_name"]
    fig, ax = plt.subplots()
    ig.plot(ig_net, **v_style_trajectory)
    ax.invert_yaxis()
    plt.show()

def dataframe_to_igraph(df):
    """
    Converts a pandas DataFrame to an igraph Graph. The DataFrame must have at least three columns:
    'TF' for the source node, 'TG' for the target node, and 'Type' indicating the nature of the interaction.
    The graph is directed, with edges going from 'TF' to 'TG', and the 'Type' attribute added to each edge.

    Args:
        df (pandas.DataFrame): The DataFrame to convert, containing 'TF', 'TG', and 'Type' columns.

    Returns:
        igraph.Graph: The resulting graph with nodes and directed edges as specified in the DataFrame.

    Raises:
        ValueError: If the required columns ('TF', 'TG', 'Type') are not present in the DataFrame.

    Examples:
        >>> data = {
        ...     'TF': ['Node1', 'Node2', 'Node3'],
        ...     'TG': ['Node3', 'Node1', 'Node2'],
        ...     'Type': ['+', '-', '+']
        ... }
        >>> df = pd.DataFrame(data)
        >>> graph = dataframe_to_igraph(df)
        >>> print("Vertices:", graph.vs['name'])
        Vertices: ['Node1', 'Node2', 'Node3']
        >>> print("Edges:", [(edge.source, edge.target) for edge in graph.es])
        Edges: [(0, 2), (1, 0), (2, 1)]
        >>> print("Edge Types:", graph.es['Type'])
        Edge Types: ['+', '-', '+']
    """
    required_columns = ['TF', 'TG', 'Type']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Create an empty directed graph
    g = Graph(directed=True)

    # Add vertices
    vertices = set(df['TF']).union(set(df['TG']))
    g.add_vertices(list(vertices))

    # Add edges with 'Type' attribute
    edges = list(zip(df['TF'], df['TG']))
    g.add_edges(edges)
    g.es['Type'] = df['Type'].tolist()

    return g

def dataframe_to_networkx(network_df): 
    """Convert gene regulatory network dataframe to networkx directed graph object.  

    Args:
        network_df (pandas.DataFrame): The DataFrame to convert, containing 'TF', 'TG', and 'Type' columns.

    Returns:
        networkx.DiGraph: networkx directed graph object describing the gene regulatory network. 
    
    Raises:
        ValueError: If the required columns ('TF', 'TG', 'Type') are not present in the DataFrame.
    """
    G = nx.DiGraph()
    network_df.index = list(range(0, network_df.shape[0])) 
    for temp_index in network_df.index: 
        if network_df.loc[temp_index, 'Type'] == "+":
            G.add_edge(network_df.loc[temp_index, "TF"],network_df.loc[temp_index, "TG"], label='activation', color='#f5428d') 
        else:
            G.add_edge(network_df.loc[temp_index, "TF"],network_df.loc[temp_index, "TG"], label='repression', color='#4c36f5')
    return G

# similar to define_states() except apply to annData, and use user-provided means and n_cells thresholds
def define_states_adata(adata, min_mean = 0.5, min_percent_cells = 0.20) -> pd.Series:
    """A quick way to define activity states of network genes for one single cell type. 

    Args:
        adata (AnnData): AnnData of one cell type. The users should subset the AnnData to only one cell type before hand. Ex: adCMP = adata[adata.obs['cell_types'] == 'CMP'].copy()
        min_mean (float, optional): minimal expression threshold for a gene to be considered as turned on. Defaults to 0.5.
        min_percent_cells (float, optional): minimal percent expression threshold for a gene to be considered as turned on. Defaults to 0.20.

    Returns:
        pd.Series: The Boolean activity profile of network genes for one single cell type. 
    """
    n_cells = adata.shape[0]
    adQuery = adata.copy()
    sc.pp.calculate_qc_metrics(adQuery, percent_top=None, inplace=True, log1p=False)
    criteria_met = (adQuery.var['mean_counts'] >= min_mean) & (adQuery.var['n_cells_by_counts']/n_cells >= min_percent_cells)
    
    # Return as pandas Series
    gene_names = adQuery.var_names if hasattr(adQuery, 'var_names') else np.arange(adQuery.X.shape[1])
    return pd.Series(criteria_met.astype(int), index=gene_names)

def infer_grn(
    cellstate_graph: nx.DiGraph,
    start_end_clusters: dict,
    adata: anndata.AnnData,
    cluster_col: str = "cluster",
    pseudoTime_col: str = 'pt',
    cutoff_percentage: float = 0.4,
    percent_exp: float = 0.3,
    ideal_edge_percent: float = 0.4,
    pseudoTime_bin: float = 0.01, 
    act_tolerance: float = 0.04, 
    GA_seed_list: list = [1, 2, 3, 4, 5], 
    init_pop_seed_list: list = [21, 22, 23, 24, 25], 
    n_cores: int = 16,
    **kwargs
) -> pd.DataFrame:
    """Wrapper function to infer gene regulatory networks using AnnData as input. 

    Args:
        cellstate_graph (nx.DiGraph): Networkx directed graph object from onesc.construct_cluster_network, onesc.construct_cluster_graph_adata or manually curated that represent cell state transitions.  
        start_end_clusters (dict): A dict of ‘start’, and ‘end’ cell states. The keys should be 'start' and 'end' and items are a list of start cell state and a list of end cell state(s). 
        adata (anndata.AnnData): The AnnData object of the single-cell dataset. The genes should be limited to dynamically expressed transcription factors or network genes. 
        cluster_col (str, optional): Column name in the AnnData.obs for the column with cluster ids or cell types. Defaults to "cluster".
        pseudoTime_col (str, optional): Column name in the AnnData.obs for the column with psuedotime information. Defaults to 'pt'.
        cutoff_percentage (float, optional): The minimum percent cut-off of the difference between highest and lowest average experssion. The cut-off for Boolean activity status for each gene is defined as "cutoff_percentage * the difference between highest and lowest cluster average expression". Defaults to 0.4.
        percent_exp (float, optional): The minimum percent expression of cells in the cell state/cluster for a gene to be considered as ON even if the average expression passes the expression threshold. Defaults to 0.3.
        ideal_edge_percent (float, optional): The ideal network density for each subnetwork. Defaults to 0.4.
        pseudoTime_bin (float, optional): The sliding psudotime bin size at which the gene expressions of the cells within are averaged forming gene expression dynamics across pseudotime. It is also the resolution at which the activity changes for the genes are recorded. If the cell density across trajectory is high, consider lowering this number for more accurate pinpoint at which genes change activity. Defaults to 0.01. 
        act_tolerance (float, optional): The pseudotime window at which two genes are considered to change at the same time in the scenerio when two changed status during cell state transition. Typically this number is 1/3-1/5 of the smallest psuedotime difference between two adjuscent cell clusters. Defaults to 0.04.
        GA_seed_list (list, optional): a list of seeds for genetic algorithm. Defaults to [1, 2, 3, 4].
        init_pop_seed_list (list, optional): a list of seeds for generating initial populations. Defaults to [20, 21, 23, 24].
        n_cores (int, optional): number of cores to run the network inference in parallel. Defaults to 16.

    Kwargs:
        selected_regulators (list, optional): The list of regulators or transcription factors. Defaults to list(). If input an empty list, then assume all genes are capable of transcriptional regulations. 
        run_parallel (bool, optional): Whether to run network inference in parallel. Defaults to True.
        num_generations (int, optional): Number of generations for genetic algorithm per gene per iteration. Defaults to 1000.
        max_iter (int, optional): Maximum number of iterations for genetic algorithm. If the fitness has not change in 3 iterations then stop early. Defaults to 10.
        num_parents_mating (int, optional): Number of parents for genetic algorithm. Defaults to 4.
        sol_per_pop (int, optional): Number of solutions to keep per generation for genetic algorithm. Defaults to 10.
        reduce_auto_reg (bool, optional): If True, remove auto activation is not needed for states satisfaction. Defaults to True.
        mutation_percent_genes (float, optional): The mutation percentage. Defaults to 25. 

    Returns:
        pd.DataFrame: Inferred GRN with majority vote from an ensemble of inferred GRNs. 
    """

    train_exp = adata.to_df().T 
    samp_tab = adata.obs.copy()

    initial_clusters = start_end_clusters['start']
    terminal_clusters = start_end_clusters['end']
    
    print("Preparing states and data for GA ...")
    lineage_cluster = extract_trajectory(cellstate_graph, initial_clusters, terminal_clusters)
    vector_thresh = find_threshold_vector(train_exp, samp_tab, cluster_col = cluster_col, cutoff_percentage = cutoff_percentage)
    lineage_time_change_dict = find_gene_change_trajectory(train_exp, samp_tab, lineage_cluster, cluster_col, pseudoTime_col, vector_thresh, pseudoTime_bin = pseudoTime_bin) 
    state_dict = define_states(train_exp, samp_tab, lineage_cluster, vector_thresh, cluster_col, percent_exp =  percent_exp)
    transition_dict = define_transition(state_dict)
    training_data = curate_training_data(state_dict, transition_dict, lineage_time_change_dict, samp_tab, cluster_id = cluster_col, pt_id = pseudoTime_col, act_tolerance = act_tolerance, **kwargs)
    corr_mat = calc_corr(train_exp)
    ideal_edge_num = round(ideal_edge_percent * corr_mat.shape[1])
    print("Starting network reconstruction with GA ...")
    grn_ensemble = create_network_ensemble(training_data, corr_mat, 
                                           ideal_edges = ideal_edge_num, GA_seed_list = GA_seed_list, 
                                           init_pop_seed_list = init_pop_seed_list, n_cores = n_cores, 
                                           **kwargs)
    print("GRN reconstruction complete.")
    inferred_grn = grn_ensemble[0]
    return inferred_grn


def simulate_parallel_adata(OneSC_simulator, init_exp_dict, network_name, perturb_dict = {}, n_cores = 2, num_runs = 10, num_sim = 1000, t_interval = 0.1, noise_amp = 0.1):
    """Running simulations using parallel, same as simulate_parallel but build a list of annData objects instead of writing to csv 

    Args:
        OneSC_simulator (onesc.OneSC_simulator): OneSC simulator object. 
        init_exp_dict (dict): the dictionary with the initial conditions for each gene. 
        network_name (str): the name of the network structure in the OneSC simulator that you want to run. 
        n_cores (int, optional): number of cores for parallel computing. Defaults to 2.
        num_runs (int, optional): number of simulations to run. Defaults to 10.
        num_sim (int, optional): number of simulation steps per simulation. Defaults to 1000.
        t_interval (float, optional): the size of the simulation step. Defaults to 0.1.
        noise_amp (float, optional): noise amplitude. Defaults to 0.1.
    """
    if n_cores > mp.cpu_count(): 
        warnings.warn("Maximum number of cores is " + str(mp.cpu_count()))
        n_cores = mp.cpu_count()

    pool = mp.pool.ThreadPool(n_cores)
    num_runs_list = list(range(0, num_runs)) 
    adata_list = []
    def run_parallel(i):
        np.random.seed(i)
        OneSC_simulator.simulate_exp(init_exp_dict, network_name, perturb_dict, num_sim = num_sim, t_interval = t_interval, noise_amp = noise_amp, random_seed = i)
        sim_exp = OneSC_simulator.sim_exp.copy()
        # sim_exp.to_csv(os.path.join(output_dir, str(i) + "_simulated_exp.csv"))
        adTemp = ad.AnnData(sim_exp.T)
        adTemp.obs['sim_time'] = sim_exp.columns.tolist()
        adata_list.append(adTemp)
    pool.map(run_parallel, num_runs_list)
    return adata_list



def sample_and_compile_anndatas(anndata_list, X, time_bin=None, sequential_order_column='sim_time'):
    """
    Samples X cells from each AnnData object in a list according to the specified range (time_bin) 
    and compiles the sampled cells into a new AnnData object. It adds metadata indicating the original
    AnnData index and the sim_time for each cell.

    Parameters:
    - anndata_list: List of AnnData objects to sample from.
    - X: The number of cells to sample from each AnnData object.
    - time_bin: A list containing the start and end of the range as percentages (0-100) of the total process length.
                If None, the complete range of cells is considered.
    - sequential_order_column: The name of the .obs column indicating the sequential order.

    Returns:
    - A new AnnData object containing the compiled sampled cells.
    """
    compiled_samples = []
    
    for i, adata in enumerate(anndata_list):
        if time_bin:
            start_percent, end_percent = time_bin
            total_range = adata.obs[sequential_order_column].max() - adata.obs[sequential_order_column].min()
            start_absolute = adata.obs[sequential_order_column].min() + total_range * (start_percent / 100)
            end_absolute = adata.obs[sequential_order_column].min() + total_range * (end_percent / 100)
            cells_in_range = adata[(adata.obs[sequential_order_column] >= start_absolute) & (adata.obs[sequential_order_column] <= end_absolute)]
        else:
            cells_in_range = adata
        
        if len(cells_in_range) > X:
            sampled_cells = cells_in_range[np.random.choice(cells_in_range.shape[0], X, replace=False), :]
        else:
            sampled_cells = cells_in_range
        
        sampled_cells.obs['origin_adata'] = i
        sampled_cells.obs['sim_time'] = sampled_cells.obs[sequential_order_column]
        
        compiled_samples.append(sampled_cells)
    
    compiled_adata = ad.concat(compiled_samples, axis=0)
    compiled_adata.obs_names_make_unique()

    return compiled_adata







