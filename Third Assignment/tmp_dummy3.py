import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import logging
from dataclasses import dataclass, field

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

def remove_random_nodes(G, chunk):
    node_list = list(G.nodes())
    if len(node_list) > chunk: rr = random.sample(node_list, k=chunk)
    for node in rr:
        G.remove_node(node)
    print("nodes removed")
    #else: rr = node_list[0]
    #if G.has_node(rr):  #TODO: while(!G.has_node(-1)) to create loop until a node is found
     #   print("node selected: ", rr)
      #  G.remove_node(rr)

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
    #TODO: more metrics: avg_centrality, avg_degree, number_of_edges

def remr(G, pos, options, chunk):
    print("Removing random node...")
    remove_random_nodes(G, chunk)
    print("nodes: ", list(G.nodes()))
    plot_dummy(G, pos, options)

def get_giant_component(G):
    cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(cc)
    return subG
    #return [G.subgraph(c).copy for c in nx.connected_components(G)]

@dataclass
class GraphMetrics:
    """Class for keeping track of the changes in the graph as the attacks follow"""
    size_of_GC: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    diameter: list = field(default_factory=list)
    degree_distribution: list = field(default_factory=list)
    avg_centrality: list = field(default_factory=list)

    def add_new_cycle(self, soGC, edges, diam, dd, avcen): #node removed, new metrics are stored
        self.size_of_GC.append(soGC)
        self.edges.append(edges)
        self.diameter.append(diam)
        self.degree_distribution.append(dd)
        self.avg_centrality.append(avcen)

def get_avg_degree(G):
    return sum([d for n, d in G.degree()])/G.number_of_nodes()

def get_avg_centrality(G):
    return sum(nx.degree_centrality(G).values())/G.number_of_nodes()

def get_avg_betweenness(G):
    return sum(nx.betweenness_centrality(G).values())/G.number_of_nodes()

def get_avg_closeness(G):
    return sum(nx.closeness_centrality(G).values())/G.number_of_nodes()

def tmp():
    #inputs required: dataset, percentage_of_nodes
    percentage = 0.1
    logging.basicConfig(level=logging.INFO)
    
    dataset = "../tech-pgp/tech-pgp.edges"
    #dataset = "../tech-routers-rf/tech-routers-rf.mtx"
    G = open_dataset(dataset)
    chunk = int(percentage*len(G.nodes()))
    print("nodes: ", G.number_of_nodes())
    print("edges: ", G.number_of_edges())
    print("chunk: ", chunk)
    #G, pos, options = dummy_dataset()
    #print("nodes: ", list(G.nodes()))
    #plot_dummy(G, pos, options)
    #print("diameter: ", nx.diameter(G))
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
    #subG = get_giant_component(G)
    #get_metrics(subG)
    #plot_dummy(G, pos, options)
    #print(max(nx.connected_components(G), key=len))

    #loop for graph size for each attack
    #attacks: random, target (node degree?), betweenness, closeness, pagerank
    graph_size = G.number_of_nodes()
    R = G.copy() #graph for random attacks
    T = G.copy() #graph for target attacks
    B = G.copy() #graph for betweenness attacks
    C = G.copy() #graph for closeness attacks
    P = G.copy() #graph for pagerank attacks

    #random, target, betweenness, closeness, pagerank
    list_GM  = [GraphMetrics(), GraphMetrics(), GraphMetrics(), GraphMetrics(), GraphMetrics()] 
    intervals = []
    num_steps = int(graph_size/chunk)-1
    print("num steps:", num_steps)
    for x in range(num_steps): #leave some nodes remaining
        print(x)
        remove_random_nodes(R, chunk)
        subRGM = get_giant_component(R)
        #remove_highest_degree_node(T)
        #subTGM = get_giant_component(T)
        list_GM[0].add_new_cycle(len(subRGM), R.number_of_edges(), nx.diameter(subRGM), get_avg_degree(R), get_avg_centrality(R))
        #list_GM[1].add_new_cycle(len(subTGM), T.number_of_edges(), nx.diameter(subTGM), get_avg_degree(T), get_avg_centrality(T))
        intervals.append(chunk*x)
    
    plt.plot(intervals, list_GM[0].diameter, marker='o')
    plt.xlabel("Nodes removed")
    plt.ylabel("Diameter")
    plt.title("Evolution of diameter")
    plt.tight_layout()
    plt.show()
tmp()