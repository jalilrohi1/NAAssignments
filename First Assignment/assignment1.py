import argparse
import matplotlib.pyplot as plt
import networkx as nx
import logging
import numpy as np
import powerlaw
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


def main(verbose: bool):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    metrics = ["Number of nodes", "Number of edges", "Lowerbound Diameter", "Minimum Degree", "Average degree", "Highest Degree", 
           "Average Centrality", "Top Centrality", "Transitivity", "Average Number of Triangles", "Highest Number of Triangles", 
           "Density", "K-Core", "K-Core Minimum Degree"]
    values = []
    top_nodes = [] #TODO

    print("Opening dataset...")
    f = open("../inf-italy-osm/inf-italy-osm-cleaned.edges", "r")
    G = nx.read_weighted_edgelist(f, nodetype=int)
    f.close()

    print("Calculating metrics...")

    logging.info("Calculating nodes...")
    values.append(G.number_of_nodes())
    logging.info("Calculating edges...")
    values.append(G.number_of_edges())
    #print("diameter: ", nx.diameter(G)) #too slow
    logging.info("Calculating lower bound diameter...")
    values.append(nx.approximation.diameter(G))

    logging.info("Calculating degrees...")
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    values.append(max(degree_sequence))
    values.append(min(degree_sequence))
    avg_degree = sum(degree_sequence)/G.number_of_nodes()
    values.append(avg_degree)

    logging.info("Calculating centrality...")
    centrality = nx.degree_centrality(G)
    centrality_avg = sum(centrality.values())/G.number_of_nodes()
    sorted_c = sorted(centrality.items(), key=lambda x:x[1], reverse=True)
    values.append(centrality_avg)
    values.append(sorted_c[0][1])

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

    logging.info("Calculating density...")
    values.append(nx.density(G))

    logging.info("Calculating kcore...")
    KC = nx.k_core(G)
    values.append(KC)
    values.append(min(sorted((d for n, d in KC.degree()), reverse=True)))

    print(table_results(metrics, values))

    plt.figure(1)
    ax1 = plt.subplot()
    ax1.bar(*np.unique(degree_sequence, return_counts=True))
    ax1.set_title("Degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")
    plt.show()

    plt.figure(2)
    plt.plot(degree_sequence, 'b-')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action='store_true')
    cli_args = parser.parse_args()
    main(cli_args.v)
    #xmin = powerlaw.find_xmin(degree_sequence)
    """
    fit = powerlaw.Fit(degree_sequence)
    fit.distribution_compare('power_law', 'lognormal')
    fig4 = fit.plot_ccdf(linewidth=3, color='black')
    fit.power_law.plot_ccdf(ax=fig4, color='r', linestyle='--') #powerlaw
    fit.lognormal.plot_ccdf(ax=fig4, color='g', linestyle='--') #lognormal
    fit.stretched_exponential.plot_ccdf(ax=fig4, color='b', linestyle='--') #stretched_exponential
    plt.show()

    """

"""
TODO:
- random or scale free network https://en.wikipedia.org/wiki/Scale-free_network
- important nodes (we can use centrality, number of triangles, max degree (gotta get the corresponding node though))
- assortative (how?)
- look for more questions in the slides

INFO OBTAINED:
- it is connected (lowest degree is 1)
- max degree is 9
- most common degree is 2
- avg centrality is 3.18e-07
- node 1489103 has highest centrality with 1.36e-06
- transitivity is 0.00269
- avg number of triangles is 0.00332
- node 1013347 has highest number of triangles with 5
"""