import pandas as pd
import networkx as nx
import torch
import numpy as np
from model.utils import encode_onehot 
from model.utils import frac_mat_power 

def load_data(path, dataset, train=200, val=300, test=1000):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    
    # Load data
    content = pd.read_csv('{}{}.content'.format(path, dataset), sep='\t', header=None)
    cites = pd.read_csv('{}{}.cites'.format(path, dataset), sep='\t', header=None)
    
    # Construct graph
    G = nx.Graph(name = 'G')

    # Create nodes
    for i in content[0]:
        G.add_node(i, name=i)

    # Create edges
    for e in cites.to_numpy():
        G.add_edge(e[1], e[0])

    #See graph info
    print('Graph Info:\n', nx.info(G))
    
    #Get the Adjacency Matrix (A) and Node Features Matrix (X) as numpy array
    A = torch.FloatTensor(nx.adjacency_matrix(G).todense())
    X = content.drop([0, 1434], axis = 1).to_numpy()
    y = encode_onehot(content[1434].to_numpy())

    idx_train = range(train)
    idx_val = range(train, train+val)
    idx_test = range(train+val, train+val+test)

    I = torch.eye(A.shape[0]) #create Identity Matrix of A
    A_hat = A + I #add self-loop to A
    D = torch.diag(torch.sum(A_hat, axis=0)) #create Degree Matrix of A
    D_half_norm = frac_mat_power(D, -0.5) #calculate D to the power of -0.5
    A = D_half_norm.mm(A_hat).mm(D_half_norm)

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