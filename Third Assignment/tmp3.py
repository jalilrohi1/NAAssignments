import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import logging
from dataclasses import dataclass, field

#TODO: functions
"""
plot_metric(x, y, xlabel, ylabel, figsize)
betweenness_selection(G)
closeness_selection(G)
pagerank_selection(G)
"""

def generate_graphs(num_nodes, density, probability, seed):
    logging.info("Generating Barabasi-Albert graph...")
    barabasi = nx.barabasi_albert_graph(num_nodes, density, seed)
    logging.info("Generating Barbell graph...")
    barbell = nx.barbell_graph(int(num_nodes/2), int(num_nodes/2))
    logging.info("Generating Circular Ladder graph...")
    circular_ladder = nx.circular_ladder_graph(num_nodes)
    #logging.info("Generating Dorogovtsev Goltsev Mendes graph...")
    #dgm = nx.dorogovtsev_goltsev_mendes_graph(num_nodes) #too heavy
    logging.info("Generating Newman-Watts-Strogatz graph...")
    nws = nx.newman_watts_strogatz_graph(num_nodes, density, probability, seed)
    return barabasi, barbell, circular_ladder, nws

def plot_graph(G): #this and other functions could be moved in a shared utility library
    nx.draw(G)
    plt.show()

def open_dataset(dataset):
    print("Opening dataset...")
    f = open(dataset, "r")
    G = nx.read_edgelist(f, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    f.close()
    return G

def get_degree_distribution(G):
    return sorted((d for n, d in G.degree()), reverse=True)

def plot_degree_distribution(G, graph_name):
    degree_counts = np.bincount(get_degree_distribution(G))
    degrees = np.arange(len(degree_counts))
    title = graph_name + " - Degree Distribution"
    plt.plot(degrees, degree_counts, 'r-')
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title(title)
    plt.show()

def attack(G:nx.Graph, r_nodes: list) -> None:
    """Remove nodes in the r_nodes list from the graph G
    """
    for node in r_nodes:
        if node in G:
            G.remove_node(node)

def select_random_nodes(G:nx.Graph, chunk:int) -> list:
    """Get list of random nodes"""
    node_list = list(G.nodes())
    targets = []
    if len(node_list) > chunk: targets = random.sample(node_list, k=chunk)
    return targets

def select_highest_degree_nodes(G:nx.Graph, chunk:int) -> list:
    """Select nodes with the highest degree"""
    dg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    targets = []
    if len(list(G.nodes())) > chunk:
        for x in range(chunk):
            targets.append(dg[x][0])
    return targets

@dataclass
class GraphMetrics:
    """Class for keeping track of the changes in the graph as the attacks follow"""
    size_of_GC: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    diameter: list = field(default_factory=list)
    degree_distribution: list = field(default_factory=list)
    avg_centrality: list = field(default_factory=list)

    def add_new_cycle(self, soGC, edges, diam, dd, avcen):
        """Save the metrics of the current attack cycle"""
        self.size_of_GC.append(soGC)
        self.edges.append(edges)
        self.diameter.append(diam)
        self.degree_distribution.append(dd)
        self.avg_centrality.append(avcen)

def get_giant_component(G: nx.Graph) -> nx.Graph:
    cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(cc)
    return subG

def get_avg_degree(G):
    return sum([d for n, d in G.degree()])/G.number_of_nodes()

def get_avg_centrality(G):
    return sum(nx.degree_centrality(G).values())/G.number_of_nodes()

def get_avg_betweenness(G):
    return sum(nx.betweenness_centrality(G).values())/G.number_of_nodes()

def get_avg_closeness(G):
    return sum(nx.closeness_centrality(G).values())/G.number_of_nodes()

def main(dataset: str, num_nodes: int, density: int, probability: float, seed: int, percentage: float, 
         verbose: bool, plot: bool, save: bool):
    
    if verbose:
        logging.basicConfig(level=logging.INFO)

    G = open_dataset(dataset)
    barabasi, barbell, circular_ladder, nws = generate_graphs(num_nodes, density, probability, seed)

    #plotting degrees of generated and loaded graphs
    if plot:
        plot_degree_distribution(G, dataset)
        plot_degree_distribution(barabasi, "Barabasi-Albert")
        plot_degree_distribution(barbell, "Barbell")
        plot_degree_distribution(circular_ladder, "Circular Ladder")
        plot_degree_distribution(nws, "Newman-Watts-Strogatz")

    #TODO: refactor, create setup() method
    #parameters for the attacks on the Barabasi-Alber model
    chunk = int(percentage*len(barabasi.nodes()))
    #maybe create list of graph copies?
    R = barabasi.copy() #graph for random attacks
    D = barabasi.copy() #graph for degree attacks
    B = barabasi.copy() #graph for betweenness attacks
    C = barabasi.copy() #graph for closeness attacks
    P = barabasi.copy() #graph for pagerank attacks
    #attacks on barabasi model
    list_GM  = [GraphMetrics(), GraphMetrics(), GraphMetrics(), GraphMetrics(), GraphMetrics()] 
    intervals = []
    num_steps = int(num_nodes/chunk)-1

    #maybe even these one can be refactored
    for x in range(num_steps): #leave some nodes remaining
        logging.info("new cycle: %s", x)
        rr_nodes = select_random_nodes(R, chunk)
        rd_nodes = select_highest_degree_nodes(R, chunk)
        #rb_nodes = select_highest_betweenness_nodes(R, chunk)
        #rc_nodes = select_highest_closeness_nodes(R, chunk)
        #rp_nodes = select_highest_pagerank_nodes(R, chunk)
        
        attack(R, rr_nodes)
        attack(D, rd_nodes)
        #attack(B, rb_nodes)
        #attack(C, rc_nodes)
        #attack(P, rp_nodes)

        subRGM = get_giant_component(R)
        subDGM = get_giant_component(D)
        #subBGM = get_giant_component(B)
        #subCGM = get_giant_component(C)
        #subPGM = get_giant_component(P)

        list_GM[0].add_new_cycle(len(subRGM), R.number_of_edges(), nx.diameter(subRGM), get_avg_degree(R), get_avg_centrality(R))
        list_GM[1].add_new_cycle(len(subDGM), D.number_of_edges(), nx.diameter(subDGM), get_avg_degree(D), get_avg_centrality(D))
        intervals.append(chunk*x)

    #temp
    plt.plot(intervals, list_GM[0].diameter, 'r-')
    plt.xlabel("Nodes removed")
    plt.ylabel("Diameter")
    plt.title("Evolution of diameter - Random")
    plt.tight_layout()
    plt.show()

    plt.plot(intervals, list_GM[0].size_of_GC, 'r-')
    plt.xlabel("Nodes removed")
    plt.ylabel("Diameter")
    plt.title("Evolution of Giant Component - Random")
    plt.tight_layout()
    plt.show()

    plt.plot(intervals, list_GM[1].diameter, 'r-')
    plt.xlabel("Nodes removed")
    plt.ylabel("Diameter")
    plt.title("Evolution of diameter")
    plt.tight_layout()
    plt.show()

    plt.plot(intervals, list_GM[1].size_of_GC, 'r-')
    plt.xlabel("Nodes removed")
    plt.ylabel("Diameter")
    plt.title("Evolution of Giant Component - Random")
    plt.tight_layout()
    plt.show()

    #TODO: verify if they work properly (load smaller datasets?)

main("../tech-routers-rf/tech-routers-rf.mtx", 4096, 3, 0.3, 420, 0.1, True, False, False)