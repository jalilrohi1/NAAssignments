import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import logging
import os

def check_continue(color_map:list):
    """Loop control"""
    if 'red' in color_map:
        return True
    return False

def update_timelist(timelist:list):
    """Updates the time for infected nodes"""
    return [x+1 if x != None else None for x in timelist]

def get_I_nodes(color_map:list):
    """Returns indexes of infected nodes"""
    lst = []
    x = 0
    for node in color_map:
        if node == 'red':
            lst.append(x)
        x+=1
    return lst

def check_recovery(color_map:list, G:nx.Graph, node_list:list, timelist:list, d:int, q:float, lb:list):
    """Simulates recovery process"""
    logging.info("Calculating nodes that can move to R..")
    R_valid = []
    for x in range(G.number_of_nodes()):
        if timelist[x] != None:
            if timelist[x] > d:
                R_valid.append(True)
            else:
                R_valid.append(False)
        else:
            R_valid.append(False)
    logging.info(R_valid)
    #for each marked node: flip coin with probability q
    logging.info("Nodes that move to R:")
    for x in range(G.number_of_nodes()):
        r = random.uniform(0.0, 1.0)
        if R_valid[x] and r <= q:
            logging.info(node_list[x])
            color_map[x] = 'g'
            nx.set_node_attributes(G, {node_list[x]:{'labels':"R"}})
            lb = nx.get_node_attributes(G, 'labels')
            timelist[x] = None
    return lb
    

def check_spreading(color_map:list, G:nx.Graph, node_list:list, timelist:list, p:float, lb:list):
    """Simulates spreading of infection"""
    I_nodes = get_I_nodes(color_map)
    logging.info("Red nodes indexes: %s", I_nodes)
    for index in I_nodes:
        node = node_list[index]
        neighbors = [n for n in G.neighbors(node)]
        logging.info("Node: %s", node)
        logging.info("Neighbors: %s", neighbors)
        for n in neighbors:
            r = random.uniform(0.0, 1.0)
            if r <= p:
                index_n = node_list.index(n)
                if color_map[index_n] != 'g' and color_map[index_n] != 'red':
                    timelist[index_n] = 0
                    color_map[index_n] = 'red'
                    nx.set_node_attributes(G, {n:{'labels':"I"}})
                    lb = nx.get_node_attributes(G, 'labels')
    return lb

def main(dataset:str, output_dir:str, p:float, d:int, q:float, i:int, verbose:bool):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    print("Opening dataset...")
    f = open(dataset, "r")
    G = nx.read_edgelist(f, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    f.close()

    num_nodes = G.number_of_nodes()
    pos = nx.spring_layout(G)
    timelist = np.full(num_nodes, None)
    node_list = list(G.nodes())
    color_map = np.full(num_nodes, 'blue')
    labels= "S"
    nx.set_node_attributes(G, labels, "labels")
    lb = nx.get_node_attributes(G, 'labels')

    if not os.path.exists(output_dir): os.mkdir(output_dir)

    print("Drawing initial graph...")
    fig = plt.figure()
    nx.draw(G, pos, alpha=0.4, width=0.3, labels=lb, node_size=25, node_color=color_map, font_size=3)
    fig.savefig(output_dir+"step0.png")
    plt.close(fig)

    print("Infecting chosen nodes...")
    chosen = []
    for x in range(i):
        rr = random.randrange(num_nodes)
        chosen.append(node_list[rr])
        timelist[rr] = 0
        color_map[rr] = 'red'
        nx.set_node_attributes(G, {chosen[x]:{'labels':"I"}})
        lb = nx.get_node_attributes(G, 'labels')
    
    logging.info("Chosen nodes: %s", chosen)
    
    print("First infection...")
    fig = plt.figure()
    nx.draw(G, pos, alpha=0.4, width=0.3, labels=lb, node_size=25, node_color=color_map, font_size=3)
    fig.savefig(output_dir+"step1.png")
    plt.close(fig)

    #loops until all nodes are green or blue
    print("Beginning SIR iterations...")
    iterations = 0
    while(check_continue(color_map)):
        iterations+=1
        timelist = update_timelist(timelist)
        lb = check_recovery(color_map, G, node_list, timelist, d, q, lb)
        lb = check_spreading(color_map, G, node_list, timelist, p, lb)
        fig = plt.figure()
        nx.draw(G, pos, alpha=0.4, width=0.3, labels=lb, node_size=25, node_color=color_map, font_size=3)
        step = "step"+str(iterations+1)+".png"
        fig.savefig(output_dir+step)
        plt.close(fig)

    print("SIR model completed in ", iterations, " iterations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="dataset file", default="../tech-routers-rf/tech-routers-rf.mtx")
    parser.add_argument("-o", type=str, help="output directory", default="outputs/")
    parser.add_argument("-p", type=float, help="transmission probability", default=0.4)
    parser.add_argument("-d", type=int, help="duration of the infection", default=4)
    parser.add_argument("-q", type=float, help="recovery probability", default=0.3)
    parser.add_argument("-i", type=int, help="individuals initially infected", default=5)
    parser.add_argument("-v", help="verbose output", action='store_true')
    cli_args = parser.parse_args()
    main(cli_args.f, cli_args.o,cli_args.p, cli_args.d, cli_args.q, cli_args.i, cli_args.v)