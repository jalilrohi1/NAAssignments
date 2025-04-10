import argparse
import matplotlib.pyplot as plt
import networkx as nx
import logging
import numpy as np
from tabulate import tabulate
import pandas as pd
import os
import math

def open_dataset(dataset:str) -> nx.Graph:
    """Loads dataset from path"""
    print("Opening dataset...")
    f = open(dataset, "r")
    G = nx.read_edgelist(f, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    f.close()
    return G

def table_results(metrics:list[str], values:list) -> str:
    """Print data in tabulate format"""
    headers = ["Metric", "Value"]
    table = zip(metrics, values)
    return tabulate(table, headers=headers)

def append_ten(lst_source:list, lst_dest:list) -> None:
    """Add ten values from lst_source to lst_dest"""
    for x in range(10):
        lst_dest.append(lst_source[x])

def save_metrics(values:list, metrics:list[str], output_dir:str, name:str):
    """Save calculated metrics on disk as a png"""
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=[values], colLabels=metrics, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(metrics))))
    fig.tight_layout()
    fig.savefig(output_dir+name, bbox_inches="tight", pad_inches=0.5)

def save_metrics_alt(values:list, metrics:list[str], output_dir:str, name:str):
    """Alterative function to save metrics using dataframes instead of lists"""
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=values, colLabels=metrics, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(metrics))))
    fig.savefig(output_dir+name, bbox_inches="tight", pad_inches=0.5)

def k_probability(num_degree:list, num_nodes:int) -> list:
    """Returns list with the probability of each node degree"""
    res = []
    for x in range(len(num_degree)):
        res.append(num_degree[x]/num_nodes)
    return res

def power_law_fitting(degrees, k_prob:list):
    """Returns a list with the values of parameter a to the corresponding node degree"""
    p_law = []
    d_law = []
    for x in range(len(k_prob)):
        v = 1.0/k_prob[x]
        if degrees[x] != 1:
            p_law.append(math.log(v, float(degrees[x])))
            d_law.append(degrees[x])
    return p_law, d_law

def main(dataset:str, output_dir:str, full:bool, verbose:bool):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if full:
        metrics = ["Number of nodes", "Number of edges", "Connected", "Diameter", "Minimum Degree", "Average degree", "Highest Degree", 
            "Average Centrality", "Top Centrality", "Transitivity", "Average Number of Triangles", "Highest Number of Triangles", 
            "Density", "K-Core", "K-Core Minimum Degree", "Assortativity", "Average Betweenness", "Top Betweenness", 
            "Average Closeness", "Top Closeness"]
    else:
        metrics = ["Number of nodes", "Number of edges", "Connected", "Lowerbound Diameter", "Minimum Degree", "Average degree", "Highest Degree", 
            "Average Centrality", "Top Centrality", "Transitivity", "Average Number of Triangles", "Highest Number of Triangles", 
            "Density", "K-Core", "K-Core Minimum Degree", "Assortativity"]
        
    values = []
    top_degree = []
    top_centrality = []
    top_triangles = []
    if full:
        top_betweenness = []
        top_closeness = []

    G = open_dataset(dataset)

    print("Calculating metrics...")
    logging.info("Calculating nodes...")
    values.append(G.number_of_nodes())
    logging.info("Calculating edges...")
    values.append(G.number_of_edges())
    logging.info("Checking if connected...")
    values.append(nx.is_connected(G))

    if full:
        logging.info("Calculating diameter...")
        values.append(nx.diameter(G))
    else:
        logging.info("Calculating lower bound diameter...")
        values.append(nx.approximation.diameter(G))

    logging.info("Calculating degrees...")
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    values.append(min(degree_sequence))
    avg_degree = sum(degree_sequence)/G.number_of_nodes()
    values.append(avg_degree)
    values.append(max(degree_sequence))
    dg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    append_ten(dg, top_degree)

    logging.info("Calculating centrality...")
    centrality = nx.degree_centrality(G)
    centrality_avg = sum(centrality.values())/G.number_of_nodes()
    sorted_c = sorted(centrality.items(), key=lambda x:x[1], reverse=True)
    values.append(centrality_avg)
    values.append(sorted_c[0][1])
    append_ten(sorted_c, top_centrality)

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

    if full:
        logging.info("Calculating betweenness...")
        betweenness = nx.betweenness_centrality(G)
        betweenness_avg = sum(betweenness.values())/G.number_of_nodes()
        sorted_b = sorted(betweenness.items(), key=lambda x:x[1], reverse=True)
        values.append(betweenness_avg)
        values.append(sorted_b[0][1])
        append_ten(sorted_b, top_betweenness)
        
        logging.info("Calculating closeness...")
        closeness = nx.closeness_centrality(G)
        closeness_avg = sum(closeness.values())/G.number_of_nodes()
        sorted_clo = sorted(closeness.items(), key=lambda x:x[1], reverse=True)
        values.append(closeness_avg)
        values.append(sorted_clo[0][1])
        append_ten(sorted_clo, top_closeness)

    if verbose: print("\n", table_results(metrics, values))

    if not os.path.exists(output_dir): os.mkdir(output_dir)

    #degree histogram
    fig = plt.figure()
    ax1 = plt.subplot()
    amounts = np.unique(degree_sequence, return_counts=True)
    ax1.bar(*amounts)
    ax1.set_title("Degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")
    fig.savefig(output_dir+"degree-histogram.png")
    plt.close(fig)

    #degree distribution
    fig = plt.figure()
    degree_counts = np.bincount(degree_sequence)
    degrees = np.arange(len(degree_counts))
    plt.plot(degrees, degree_counts, 'r-')
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    fig.savefig(output_dir+"degree-distribution.png")
    plt.close(fig)

    #alpha-value for each degree (excluding 1)
    fig = plt.figure()
    k_prob = k_probability(amounts[1], G.number_of_nodes())
    p_law, d_law = power_law_fitting(amounts[0], k_prob)
    plt.plot(d_law, p_law, 'g-')
    plt.xlabel('Degree')
    plt.ylabel('Value of alpha')
    plt.title('p(k) = 1/k^a')
    fig.savefig(output_dir+"power-law-fitting.png")
    plt.close(fig)

    #metrics table
    save_metrics(values[0:4], metrics[0:4], output_dir, "metrics-1.png")
    save_metrics(values[4:7], metrics[4:7], output_dir, "metrics-2.png")
    save_metrics(values[7:10], metrics[7:10], output_dir, "metrics-3.png")
    save_metrics(values[10:13], metrics[10:13], output_dir, "metrics-4.png")
    save_metrics(values[13:16], metrics[13:16], output_dir, "metrics-5.png")
    if full:
        save_metrics(values[16:], metrics[16:], output_dir, "metrics-6.png")
    #top 10 nodes for specific metrics    
    df = pd.DataFrame(data={'Nodes-Degree': [x[0] for x in top_degree], 'Degree': [y[1] for y in top_degree]})
    if verbose: print("\n", df)
    save_metrics_alt(df.values, df.columns, output_dir, "top10-degree.png")
    df = pd.DataFrame(data={'Nodes-Centrality': [x[0] for x in top_centrality], 'Centrality': [y[1] for y in top_centrality]})
    if verbose: print("\n", df)
    save_metrics_alt(df.values, df.columns, output_dir, "top10-centrality.png")
    df = pd.DataFrame(data={'Nodes-Triangles': [x[0] for x in top_triangles], 'Triangles': [y[1] for y in top_triangles]})
    if verbose: print("\n", df)
    save_metrics_alt(df.values, df.columns, output_dir, "top10-triangles.png")

    if full:
        df = pd.DataFrame(data={'Nodes-Betweenness': [x[0] for x in top_betweenness], 'Betweenness': [y[1] for y in top_betweenness]})
        if verbose: print("\n", df)
        save_metrics_alt(df.values, df.columns, output_dir, "top10-betweenness.png")
        df = pd.DataFrame(data={'Nodes-Closeness': [x[0] for x in top_closeness], 'Closeness': [y[1] for y in top_closeness]})
        if verbose: print("\n", df)
        save_metrics_alt(df.values, df.columns, output_dir, "top10-closeness.png")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="dataset file", default="../inf-italy-osm/inf-italy-osm-cleaned.edges")
    parser.add_argument("-o", type=str, help="ouput directory", default="outputs/")
    parser.add_argument("-full", help="calculate slow metrics", action='store_true')
    parser.add_argument("-v", help="verbose output", action='store_true')
    cli_args = parser.parse_args()
    main(cli_args.f, cli_args.o, cli_args.full, cli_args.v)