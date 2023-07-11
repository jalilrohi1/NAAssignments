import argparse
import matplotlib.pyplot as plt
import networkx as nx
import logging
import numpy as np
from scipy.interpolate import make_interp_spline
from tabulate import tabulate

"""
basic metrics:
V edges
V nodes
X diameter (too long?)
V degree
V centrality https://networkx.org/documentation/stable/reference/algorithms/centrality.html
X betweenness https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html
X closeness https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
- brokerage https://github.com/rafguns/gefura
V transitivity https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html
V triangles https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.triangles.html
"""

def table_results(metrics, values):
    headers = ["Metric", "Value"]
    table = zip(metrics, values)
    return tabulate(table, headers=headers)

def column_results(header, values):
    return tabulate(values, headers=header)

def append_ten(lst_source, lst_dest):   #append first ten elements in source to dest
    for x in range(10):
        lst_dest.append(lst_source[x])

def main(dataset: str, verbose: bool):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    metrics = ["Number of nodes", "Number of edges", "Lowerbound Diameter", "Minimum Degree", "Average degree", "Highest Degree", 
           "Average Centrality", "Top Centrality", "Transitivity", "Average Number of Triangles", "Highest Number of Triangles", 
           "Density", "K-Core", "K-Core Minimum Degree", "Assortativity"]
    values = []
    top_degree = []
    top_centrality = []
    top_triangles = []

    print("Opening dataset...")
    f = open(dataset, "r")
    G = nx.read_edgelist(f, nodetype=int)
    f.close()

    print("Calculating metrics...")
    logging.info("Calculating nodes...")
    values.append(G.number_of_nodes())
    logging.info("Calculating edges...")
    values.append(G.number_of_edges())
    #values.append(nx.diameter(G)) #too slow
    logging.info("Calculating lower bound diameter...")
    values.append(nx.approximation.diameter(G))

    logging.info("Calculating degrees...")
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    values.append(max(degree_sequence))
    values.append(min(degree_sequence))
    avg_degree = sum(degree_sequence)/G.number_of_nodes()
    values.append(avg_degree)
    dg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    append_ten(dg, top_degree)

    logging.info("Calculating centrality...")
    centrality = nx.degree_centrality(G)
    centrality_avg = sum(centrality.values())/G.number_of_nodes()
    sorted_c = sorted(centrality.items(), key=lambda x:x[1], reverse=True)
    values.append(centrality_avg)
    values.append(sorted_c[0][1])
    append_ten(sorted_c, top_centrality)

    #print("betweenness: ", nx.betweenness_centrality(G)) #this is very slow to calculate
    #print("approximated betweenness: ", nx.betweenness_centrality(G, k=100)) #still too slow
    #print("closeness: ", nx.closeness_centrality(G)) #too slow

    logging.info("Calculating transitivity...")
    values.append(nx.transitivity(G))

    logging.info("Calculating triangles...")
    tmp = nx.triangles(G)
    avg_triangles = sum(tmp.values())/G.number_of_nodes()
    high_t = sorted(tmp.items(), key=lambda x:x[1], reverse=True)
    values.append(avg_triangles)
    values.append(high_t[0][1])
    append_ten(high_t, top_triangles)

    logging.info("Calculating density...")
    values.append(nx.density(G))

    logging.info("Calculating kcore...")
    KC = nx.k_core(G)
    values.append(KC)
    values.append(min(sorted((d for n, d in KC.degree()), reverse=True)))

    logging.info("Calculating assortativity...")
    values.append(nx.degree_assortativity_coefficient(G))

    print(table_results(metrics, values))
    
    print("\nTop 10 nodes for each metric\n")
    print(column_results(["Node", "Degree"], top_degree))
    print("\n")
    print(column_results(["Node", "Centrality"], top_centrality))
    print("\n")
    print(column_results(["Node", "Triangles"], top_triangles))

    plt.figure(1)
    ax1 = plt.subplot()
    ax1.bar(*np.unique(degree_sequence, return_counts=True))
    ax1.set_title("Degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")
    plt.show()

    #TODO: add more graphs about centrality and other metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="dataset file", default="../inf-italy-osm/inf-italy-osm-cleaned.edges")
    #TODO: add plottable flag like in assignment2.py
    parser.add_argument("-v", action='store_true')
    cli_args = parser.parse_args()
    main(cli_args.f, cli_args.v)

"""
TODO:
- random or scale free network https://en.wikipedia.org/wiki/Scale-free_network -> not random, sort of scale free?
- smoother graph -> https://www.geeksforgeeks.org/how-to-plot-a-smooth-curve-in-matplotlib/
"""