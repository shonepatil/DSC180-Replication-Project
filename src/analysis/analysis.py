import pandas as pd
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
from pyvis.network import Network
import random

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
    print('Shape of features: ', nodes.drop([0, 1434], axis=1).to_numpy().shape)
    
    # Ad hoc features
    degree = np.array([x[1] for x in list(G.degree)])
    closeness_centr = np.array([x[1] for x in nx.closeness_centrality(G).items()])
    avg_clus_coeff = nx.average_clustering(G)

    print('\nAverage Degree: ', round(np.mean(degree), 3))
    print('Average Closeness Centrality: ', round(np.mean(closeness_centr), 3))
    print('Average clustering coefficient: ' + str(avg_clus_coeff))

    plt.hist(degree, bins=40, range=(0, 40))
    plt.xlabel("Degree")
    plt.ylabel("Counts")
    plt.title("Degree Distribution")
    plt.savefig('src/analysis/graphs/degree_dist.png')

    # Label distribution
    y = nodes[1434]
    grouped = y.value_counts()
    print('\nLabel Distribution')
    print(grouped)

    ax = grouped.plot.barh()
    ax.set_xlabel("Counts")
    ax.set_ylabel("Subject")
    ax.set_title("Label Distribution")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig('src/analysis/graphs/label_dist.png')

    # Plot CORA Graph
    g = Network(height = 700, width = 1000, notebook = True, bgcolor="#222222", font_color="white")
    g.toggle_hide_edges_on_drag(False)
    g.force_atlas_2based()
    labels = nodes[[0, 1434]].set_index(0).to_dict()[1434]

    for node in list(G.nodes):
        g.add_node(node, label = labels[node])

    x = 0
    e = list(G.edges)
    random.shuffle(e)
    for edge in e:
        g.add_edge(int(edge[0]), int(edge[1]))
        x+=1
        if x == 1000:
            break

    g.show('src/analysis/graphs/papers_graph.html')
