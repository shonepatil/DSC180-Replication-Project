from operator import xor
import pandas as pd
import networkx as nx
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from node2vec import Node2Vec

import sys
sys.path.insert(0, 'src/model')
from src.model.utils import encode_onehot, frac_mat_power

def load_data(path, dataset, train, val, test, include_ad_hoc_feat=False, include_node2vec=False):
    """Load network dataset"""
    print('Loading {} dataset...'.format(dataset))
    
    # Load data
    nodes = pd.read_csv(os.path.join(os.path.dirname(__file__), '{}{}.content'.format(path, dataset)), sep='\t', header=None)
    edges = pd.read_csv(os.path.join(os.path.dirname(__file__), '{}{}.cites'.format(path, dataset)), sep='\t', header=None)

    # Construct graph
    G = nx.Graph(name = 'G')

    # Create nodes
    for i in nodes.iloc[:, 0]:
        G.add_node(i, name=i)

    # Create edges
    for e in edges.to_numpy():
        G.add_edge(e[0], e[1])

    #See graph info
    print('Graph Info:\n', nx.info(G))
    
    #Get the Adjacency Matrix (A) and Node Features Matrix (X) as numpy array
    A = torch.FloatTensor(nx.adjacency_matrix(G).todense())
    X = nodes.drop([0, 1434], axis=1).to_numpy().astype(float)
    y = encode_onehot(nodes[1434].to_numpy())

    # Include Ad-Hoc graph variables
    if include_ad_hoc_feat:
        degree = np.array([x[1] for x in list(G.degree)])
        closeness_centr = np.array([x[1] for x in nx.closeness_centrality(G).items()])
        X = np.c_[X, degree, closeness_centr].astype(float)

    idx_train = range(train)
    idx_val = range(train, train+val)
    idx_test = range(train+val, train+val+test)

    # node2vec setup
    if include_node2vec:
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
        # Embed nodes
        mod = node2vec.fit(window=10, min_count=1, batch_words=4)
        emb = np.array(mod.wv.vectors)
        X = np.c_[X, emb].astype(float)

    I = torch.eye(A.shape[0]) #create Identity Matrix of A
    A_hat = A + I #add self-loop to A
    D = torch.diag(torch.sum(A_hat, axis=0)) #create Degree Matrix of A
    D_half_norm = frac_mat_power(D, -0.5) #calculate D to the power of -0.5
    A = D_half_norm.mm(A_hat).mm(D_half_norm)

    # Standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.FloatTensor(X)

    y = torch.LongTensor(np.where(y)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('ad_hoc_features included: ', include_ad_hoc_feat)
    print('node2vec included: ', include_node2vec)
    print('Shape of A: ', A.shape)
    print('\nShape of X: ', X.shape)
    print('\nAdjacency Matrix (A):\n', A)
    print('\nNode Features Matrix (X):\n', X)
    
    return A, X, y, idx_train, idx_val, idx_test