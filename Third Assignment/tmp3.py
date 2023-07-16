from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text
import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import logging
from dataclasses import dataclass, field
import os

def generate_graphs(num_nodes:int, density:float, probability:float, seed:int) -> list[nx.Graph]:
    """Returns a list of graphs using Networkx generators.
        Graphs are: Barabasi-Albert, Barbell, Circular Ladder and Newman-Watts-Strogatz"""
    logging.info("Generating Barabasi-Albert graph...")
    barabasi = nx.barabasi_albert_graph(num_nodes, density, seed)
    logging.info("Generating Barbell graph...")
    barbell = nx.barbell_graph(int(num_nodes/2), int(num_nodes/2))
    logging.info("Generating Circular Ladder graph...")
    circular_ladder = nx.circular_ladder_graph(num_nodes)
    logging.info("Generating Newman-Watts-Strogatz graph...")
    nws = nx.newman_watts_strogatz_graph(num_nodes, density, probability, seed)
    return barabasi, barbell, circular_ladder, nws

def open_dataset(dataset:str) -> nx.Graph:
    """Loads dataset from path"""
    print("Opening dataset...")
    f = open(dataset, "r")
    G = nx.read_edgelist(f, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    f.close()
    return G

def get_degree_distribution(G:nx.Graph) -> list[int]:
    """Returns the degree distribution of a graph in list format"""
    return sorted((d for n, d in G.degree()), reverse=True)

def plot_degree_distribution(G:nx.Graph, graph_name:str, output_dir: str) -> None:
    """Save in a directory the plot of the degree distribution of the graph"""
    degree_counts = np.bincount(get_degree_distribution(G))
    degrees = np.arange(len(degree_counts))
    title = graph_name + " - Degree Distribution"
    plt.plot(degrees, degree_counts, 'r-')
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title(title)
    plt.savefig(output_dir)
    plt.close()

def attack(G:nx.Graph, r_nodes: list) -> None:
    """Remove nodes in the r_nodes list from the graph G"""
    for node in r_nodes:
        G.remove_node(node)

def select_random_nodes(G:nx.Graph, chunk:int) -> list[int]:
    """Get list of random nodes"""
    node_list = list(G.nodes())
    targets = []
    if len(node_list) > chunk: targets = random.sample(node_list, k=chunk)
    return targets

def select_highest_degree_nodes(G:nx.Graph, chunk:int) -> list[int]:
    """Select nodes with the highest degree"""
    dg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    targets = []
    if len(list(G.nodes())) > chunk:
        for x in range(chunk):
            targets.append(dg[x][0])
    return targets

def select_highest_betweenness_nodes(G:nx.Graph, chunk:int) -> list[int]:
    """Select nodes with the highest betweenness"""
    bg = sorted(nx.betweenness_centrality(G).items(), key=lambda x:x[1], reverse=True)
    targets = []
    if len(list(G.nodes())) > chunk:
        for x in range(chunk):
            targets.append(bg[x][0])
    return targets

def select_highest_closeness_nodes(G:nx.Graph, chunk:int) -> list[int]:
    """Select nodes with the highest closeness"""
    cg = sorted(nx.closeness_centrality(G).items(), key=lambda x:x[1], reverse=True)
    targets = []
    if len(list(G.nodes())) > chunk:
        for x in range(chunk):
            targets.append(cg[x][0])
    return targets

def select_highest_pagerank_nodes(G:nx.Graph, chunk:int) -> list[int]:
    """Select nodes with the highest pagerank"""
    pg = sorted(nx.pagerank(G).items(), key=lambda x:x[1], reverse=True)
    targets = []
    if len(list(G.nodes())) > chunk:
        for x in range(chunk):
            targets.append(pg[x][0])
    return targets

@dataclass
class GraphMetrics:
    """Class for keeping track of the changes in the graph as the attacks follow"""
    size_of_GC: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    diameter: list = field(default_factory=list)
    avg_degree: list = field(default_factory=list)
    avg_centrality: list = field(default_factory=list)

    def add_new_cycle(self, soGC, edges, diam, dd, avcen):
        """Save the metrics of the current attack cycle"""
        self.size_of_GC.append(soGC)
        self.edges.append(edges)
        self.diameter.append(diam)
        self.avg_degree.append(dd)
        self.avg_centrality.append(avcen)

def get_giant_component(G:nx.Graph) -> nx.Graph:
    cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(cc)
    return subG

def get_avg_degree(G:nx.Graph) -> float:
    return sum([d for n, d in G.degree()])/G.number_of_nodes()

def get_avg_centrality(G:nx.Graph) -> float:
    return sum(nx.degree_centrality(G).values())/G.number_of_nodes()

def get_avg_betweenness(G:nx.Graph) -> float:
    return sum(nx.betweenness_centrality(G).values())/G.number_of_nodes()

def get_avg_closeness(G:nx.Graph) -> float:
    return sum(nx.closeness_centrality(G).values())/G.number_of_nodes()

def setup_attacks(G:nx.Graph, percentage:float) -> tuple[int, int, list[nx.Graph], list[GraphMetrics]]:
    """Setup the resources for the attacks"""
    chunk = int(percentage*len(G.nodes()))
    num_steps = int(G.number_of_nodes()/chunk)-1
    graph_copies = [G.copy(), G.copy(), G.copy(), G.copy(), G.copy()]
    graph_metrics = [GraphMetrics(), GraphMetrics(), GraphMetrics(), GraphMetrics(), GraphMetrics()]
    return chunk, num_steps, graph_copies, graph_metrics

def create_fig(intervals:list[int], y_random, y_degree, y_bet, y_close, y_page, y_label:str, title:str) -> tuple[Figure, Legend, Text]:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(intervals, y_random, 'r-', label='Random')
    ax.plot(intervals, y_degree, 'b-', label='Degree')
    ax.plot(intervals, y_bet, 'g-', label='Betweenness')
    ax.plot(intervals, y_close, 'c-', label='Closeness')
    ax.plot(intervals, y_page, 'y-', label='Pagerank')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    text = ax.text(-0.2,1.05, "", transform=ax.transAxes)
    ax.set_xlabel("Nodes removed")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return fig, lgd, text

def save_metrics(intervals:list[int], list_GM:list[GraphMetrics], output_dir:str) -> None:
    """Save measured metrics on the graph"""
    fig1, lgd1, text1 = create_fig(intervals, list_GM[0].size_of_GC, list_GM[1].size_of_GC, list_GM[2].size_of_GC, 
                                   list_GM[3].size_of_GC, list_GM[4].size_of_GC, "Size of Giant Component", 
                                   "Evolution of Giant Component")
    
    fig2, lgd2, text2 = create_fig(intervals, list_GM[0].edges, list_GM[1].edges, list_GM[2].edges, 
                                   list_GM[3].edges, list_GM[4].edges, "Number of edges", 
                                   "Number of edges across the attacks")
    
    fig3, lgd3, text3 = create_fig(intervals, list_GM[0].diameter, list_GM[1].diameter, list_GM[2].diameter, 
                                   list_GM[3].diameter, list_GM[4].diameter, "Diameter", 
                                   "Evolution of Diameter")
    
    fig4, lgd4, text4 = create_fig(intervals, list_GM[0].avg_degree, list_GM[1].avg_degree, list_GM[2].avg_degree, 
                                   list_GM[3].avg_degree, list_GM[4].avg_degree, "Average Degree", 
                                   "Average Degree across the attacks")
    
    fig5, lgd5, text5 = create_fig(intervals, list_GM[0].avg_centrality, list_GM[1].avg_centrality, list_GM[2].avg_centrality, 
                                   list_GM[3].avg_centrality, list_GM[4].avg_centrality, "Average Centrality", 
                                   "Average Centrality across the attacks")
    
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    logging.info("Plots will be saved in: %s", output_dir)

    fig1.savefig(output_dir+"/giant-component.png", bbox_extra_artists=(text1, lgd1), bbox_inches='tight')
    plt.close(fig1)
    fig2.savefig(output_dir+"/edges.png", bbox_extra_artists=(text2, lgd2), bbox_inches='tight')
    plt.close(fig2)
    fig3.savefig(output_dir+"/diameter.png", bbox_extra_artists=(text3, lgd3), bbox_inches='tight')
    plt.close(fig3)
    fig4.savefig(output_dir+"/avg-degree.png", bbox_extra_artists=(text4, lgd4), bbox_inches='tight')
    plt.close(fig4)
    fig5.savefig(output_dir+"/avg-centrality.png", bbox_extra_artists=(text5, lgd5), bbox_inches='tight')
    plt.close(fig5)

def launch_attacks(G:nx.Graph, percentage:float, output_dir:str, graph_name:str) ->None:
    """Main attack function"""
    chunk, num_steps, list_graphs, list_GM = setup_attacks(G, percentage)
    intervals = []
    targets = [None, None, None, None, None]
    logging.info("Starting attacks on %s...", graph_name)
    for x in range(num_steps):
        #get list of target nodes
        targets[0] = select_random_nodes(list_graphs[0], chunk)
        targets[1] = select_highest_degree_nodes(list_graphs[1], chunk)
        targets[2] = select_highest_betweenness_nodes(list_graphs[2], chunk)
        targets[3] = select_highest_closeness_nodes(list_graphs[3], chunk)
        targets[4] = select_highest_pagerank_nodes(list_graphs[4], chunk)
        
        #loop for list_graphs: attack, get_giant_component, list_GM.append
        for k in range(5):
            attack(list_graphs[k], targets[k])
            gc = get_giant_component(list_graphs[k])
            list_GM[k].add_new_cycle(len(gc), list_graphs[k].number_of_edges(), nx.diameter(gc), get_avg_degree(list_graphs[k]), get_avg_centrality(list_graphs[k]))

        intervals.append(chunk*x)
    save_metrics(intervals, list_GM, output_dir+graph_name)

def main(dataset:str, num_nodes:int, density:int, probability:float, seed:int, percentage:float,
         verbose:bool, output_dir:str) -> None:

    if verbose:
        logging.basicConfig(level=logging.INFO)

    G = open_dataset(dataset)
    barabasi, barbell, circular_ladder, nws = generate_graphs(num_nodes, density, probability, seed)

    if not os.path.exists(output_dir): os.mkdir(output_dir)
    dest = output_dir + "degree_distribution"
    if not os.path.exists(dest): os.mkdir(dest)

    print("Calcuating degree distributions...")
    logging.info("Outputs will be saved in %s", dest)
    plot_degree_distribution(barabasi, "Barabasi-Albert", dest+"/barabasi-albert.png")
    plot_degree_distribution(barbell, "Barbell", dest+"/barbell.png")
    plot_degree_distribution(circular_ladder, "Circular Ladder", dest+"/circular-ladder.png")
    plot_degree_distribution(nws, "Newman-Watts-Strogatz", dest+"/newman-watts-strogatz.png")
    plot_degree_distribution(G, dataset, dest+"/loaded-graph.png")

    launch_attacks(barabasi, percentage, output_dir, "barabasi-albert")
    launch_attacks(barbell, percentage, output_dir, "barbell")
    launch_attacks(circular_ladder, percentage, output_dir, "circular-ladder")
    launch_attacks(nws, percentage, output_dir,"newman-watts-strogatz")
    launch_attacks(G, percentage, output_dir, "loaded-graph")

main("../tech-routers-rf/tech-routers-rf.mtx", 2000, 4, 0.3, 420, 0.1, True, "outputs/") #let's use this one for my sanity's sake
#main("/content/drive/MyDrive/Colab Notebooks/tech-pgp.edges", 4096, 3, 0.3, 420, 0.1, True, False, False) #almost 21 minutes run time execution for centrality
                                                                                                          #that's a bit too much considering it would be 5x in the real execution
                                                                                                          #going for the 3+ hours of run time for 1 graph... yeah no