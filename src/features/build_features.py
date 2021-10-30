from operator import xor
import pandas as pd
import networkx as nx
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, 'src/model')
from utils import encode_onehot, frac_mat_power

def load_data(path, dataset, train, val, test):
    """Load network dataset"""
    print('Loading {} dataset...'.format(dataset))
    
    # Load data
    nodes = pd.read_csv(os.path.join(os.path.dirname(__file__), '{}{}target.csv'.format(path, dataset)), sep=',')
    edges = pd.read_csv(os.path.join(os.path.dirname(__file__), '{}{}edges.csv'.format(path, dataset)), sep=',')
    
    # Construct graph
    G = nx.Graph(name = 'G')

    # Create nodes
    for i in nodes.iloc[:, -1]:
        G.add_node(i, name=i)

    # Create edges
    for e in edges.to_numpy():
        G.add_edge(e[0], e[1])

    #See graph info
    print('Graph Info:\n', nx.info(G))
    
    #Get the Adjacency Matrix (A) and Node Features Matrix (X) as numpy array
    A = torch.FloatTensor(nx.adjacency_matrix(G).todense())
    X = nodes.iloc[:, [1, 3, 4]].to_numpy().astype(float)
    y = encode_onehot(nodes.iloc[:, 2].to_numpy())

    idx_train = range(train)
    idx_val = range(train, train+val)
    idx_test = range(train+val, train+val+test)

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

    print('Shape of A: ', A.shape)
    print('\nShape of X: ', X.shape)
    print('\nAdjacency Matrix (A):\n', A)
    print('\nNode Features Matrix (X):\n', X)
    
    return A, X, y, idx_train, idx_val, idx_test