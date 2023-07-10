import argparse
import matplotlib.pyplot as plt
import networkx as nx
import logging
import numpy as np
from tabulate import tabulate
import random

def main(dataset: str, p: float, d: int, q: float, i: int, verbose: bool):
    print("Opening dataset...")
    f = open(dataset, "r")
    G = nx.read_edgelist(f, nodetype=int)
    f.close()

    num_nodes = G.number_of_nodes()
    pos = nx.spring_layout(G)
    labels=[]
    nx.set_node_attributes(G, labels, "labels")
    labels.append("S")
    lb = nx.get_node_attributes(G, 'labels')
    
    color_map = []
    for node in G: color_map.append('blue')

    #select i nodes
    nodes = []
    for n in range(i): nodes.append(random.randrange(num_nodes))

    #change labels to I
    #change color to red
    for n in range(i):
        print(list(G.nodes())[nodes[n]])
        print(nodes[n]) 
        color_map[nodes[n]] = 'red'
        #G.nodes[n]['labels'] = "I"
        #nx.set_node_attributes(G, {nodes[n]:"I"}, 'labels')
        nx.set_node_attributes(G, {nodes[n]:{'labels':"I"}})
    
    for x in range(num_nodes):
        tmp = str(list(G.nodes(data=True))[x])
        if "I" in tmp: print(tmp)
        #if 'labels' in list(G.nodes(data=True))[x]:
        #print()
    #set infection time to 0
    infection_cycle = 0
    #loop infection/recovery


    #test with different parameters
    
    print("Drawing graph...")
    nx.draw(G, pos, alpha=0.4, width=0.3, labels=lb, node_size=50, node_color=color_map, font_size=3)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="dataset file", default="../inf-italy-osm/inf-italy-osm-cleaned.edges")
    parser.add_argument("-p", type=float, help="transmission probability", default=0.1)
    parser.add_argument("-d", type=int, help="duration of the infection", default=4)
    parser.add_argument("-q", type=float, help="recovery probability", default=0.3)
    parser.add_argument("-i", type=int, help="individuals initially infected", default=5)
    parser.add_argument("-v", action='store_true')
    cli_args = parser.parse_args()
    main(cli_args.f, cli_args.p, cli_args.d, cli_args.q, cli_args.i, cli_args.v)