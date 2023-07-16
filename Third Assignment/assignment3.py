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

def main():
    logging.basicConfig(level=logging.INFO)
    
    dataset = "../tech-pgp/tech-pgp.edges"
    G = open_dataset(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="dataset file", default="../tech-pgp/tech-pgp.edges")
    parser.add_argument("-plot", action='store_true')
    parser.add_argument("-v", action='store_true')
    cli_args = parser.parse_args()
    #main(cli_args.f, cli_args.p, cli_args.d, cli_args.q, cli_args.i, cli_args.plot, cli_args.v)
    main()