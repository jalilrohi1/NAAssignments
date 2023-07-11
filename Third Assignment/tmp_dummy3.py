import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import logging

def open_dataset(dataset):
    print("Opening dataset...")
    f = open(dataset, "r")
    G = nx.read_edgelist(f, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    f.close()
    return G

def dummy_dataset():
    G = nx.Graph()
    G.add_edge(3, 4)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 5)
    G.add_edge(2, 3)
    G.add_edge(4, 5)
    seed = 420
    pos = nx.spring_layout(G, seed=seed)

    options = {
        "font_size": 36,
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 5,
        "width": 5,
    }
    return G, pos, options

def remove_random_node(G):
    node_list = list(G.nodes())
    rr = random.randrange(min(node_list), max(node_list))
    if G.has_node(rr):
        print("node selected: ", rr)
        G.remove_node(rr)

def remove_highest_degree_node(G):
    dg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    target = max(dg)[0]
    print("node selected: ", target)
    G.remove_node(target)

def plot_dummy(G, pos, options):
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()

def get_metrics(G):
    #cc = max(nx.connected_components(G), key=len)
    print("Size of Giant Component: ", len(G))
    print("diameter: ", nx.diameter(G))
    #TODO: more metrics: avg_centrality, avg_degree, number_of_edges, 

def remr(G, pos, options):
    print("Removing random node...")
    remove_random_node(G)
    print("nodes: ", list(G.nodes()))
    plot_dummy(G, pos, options)

def get_giant_component(G):
    cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(cc)
    return subG
    #return [G.subgraph(c).copy for c in nx.connected_components(G)]

def tmp():
    logging.basicConfig(level=logging.INFO)
    
    dataset = "../tech-pgp/tech-pgp.edges"
    G = open_dataset(dataset)
    
    G, pos, options = dummy_dataset()
    print("nodes: ", list(G.nodes()))
    plot_dummy(G, pos, options)
    print("diameter: ", nx.diameter(G))
    """
    remr(G, pos, options)
    subG = get_giant_component(G)
    get_metrics(subG)

    remr(G, pos, options)
    subG = get_giant_component(G)
    get_metrics(subG)
    """
    """
    print("Removing highest degree node...")
    remove_highest_degree_node(G)
    print("nodes: ", list(G.nodes()))
    plot_dummy(G, pos, options)

    """
    G.remove_node(3)
    G.remove_node(1)
    G.remove_node(4)
    subG = get_giant_component(G)
    get_metrics(subG)
    plot_dummy(G, pos, options)
    #print(max(nx.connected_components(G), key=len))

tmp()