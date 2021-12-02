import pandas as pd
import networkx as nx
import numpy as np
import os

def compute_aggregates(path, dataset):
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

    # See graph info
    print('Graph Info:\n', nx.info(G))
    
    # Ad hoc features
    degree = np.array([x[1] for x in list(G.degree)])
    closeness_centr = np.array([x[1] for x in nx.closeness_centrality(G).items()])
    print('Average Degree: ', round(np.mean(degree), 3))
    print('Average Closeness Centrality: ', round(np.mean(closeness_centr), 3))
