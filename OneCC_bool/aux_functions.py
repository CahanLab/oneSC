from sklearn.decomposition import PCA 
import umap
import numpy as np 
import pandas as pd
 

def UMAP_embedding_train(train_exp): 
    """Trains a UMAP embedder 

    Args:
        train_exp ([pandas.DataFrame]): training data 
    """
    training_obs = dict()
    training_obs['feature_genes'] = np.array(train_exp.index)
    train_exp = train_exp.T

    my_PCA = PCA()
    my_PCA.fit(train_exp)

    training_obs['PCA'] = my_PCA

    PCA_features = my_PCA.transform(train_exp)
    if PCA_features.shape[1] < 9: 
        n_PC = PCA_features.shape[1]
    else: 
        n_PC = 9
    PCA_features = pd.DataFrame(PCA_features, index = train_exp.index)
    PCA_features = PCA_features.iloc[:, 0:n_PC]

    my_UMAP = umap.UMAP().fit(PCA_features)
    training_obs['UMAP'] = my_UMAP

    return(training_obs)

def UMAP_embedding_apply(train_obs, query_exp): 
    feature_genes = train_obs['feature_genes']
    my_PCA = train_obs['PCA']
    my_UMAP = train_obs['UMAP']

    query_exp = query_exp.loc[feature_genes, :]
    query_exp = query_exp.T 
    
    PCA_features = my_PCA.transform(query_exp)
    if PCA_features.shape[1] < 9: 
        n_PC = PCA_features.shape[1]
    else: 
        n_PC = 9
    PCA_features = pd.DataFrame(PCA_features, index = query_exp.index)
    PCA_features = PCA_features.iloc[:, 0:n_PC]

    UMAP_matrix = my_UMAP.transform(PCA_features)
    UMAP_matrix = pd.DataFrame(UMAP_matrix, index = PCA_features.index, columns = ['UMAP_1', 'UMAP_2'])
    return UMAP_matrix