import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

def check_continue(color_map):
    if 'red' in color_map:
        return True
    return False

def update_timelist(timelist):
    return [x+1 if x != None else None for x in timelist]

#returns red nodes indexes
def get_I_nodes(color_map):
    lst = []
    x = 0
    for node in color_map:
        if node == 'red':
            lst.append(x)
        x+=1
    return lst

def check_recovery(color_map, G, node_list, timelist, d, q, lb):
    #get all nodes with timelist > d
    R_valid = []
    for x in range(G.number_of_nodes()):
        if timelist[x] != None:
            if timelist[x] > d:
                R_valid.append(True)
            else:
                R_valid.append(False)
        else:
            R_valid.append(False)
    print("R valid: ", R_valid)
    #for each marked node: flip coin with probability q
    for x in range(G.number_of_nodes()):
        r = random.uniform(0.0, 1.0)
        if R_valid[x] and r <= q:
            print(node_list[x])
            color_map[x] = 'g'
            nx.set_node_attributes(G, {node_list[x]:{'labels':"R"}})
            lb = nx.get_node_attributes(G, 'labels')
            timelist[x] = None
    return lb
    

def check_spreading(color_map, G, node_list, p, lb):
    I_nodes = get_I_nodes(color_map)
    print("red nodes indexes: ", I_nodes)
    for index in I_nodes:
        node = list(G.nodes)[index]
        neighbors = [n for n in G.neighbors(node)]
        #TODO: filter R neighbors
        print(neighbors)
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

G = nx.Graph()
G.add_edge(3, 4)
G.add_edge(1, 2)
G.add_edge(1, 3)
G.add_edge(1, 5)
G.add_edge(2, 3)
G.add_edge(4, 5)

# explicitly set positions
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
timelist = np.full(5, None)
print(list(G.nodes()))
rr = random.randrange(5)
chosen = list(G.nodes)[rr]
timelist[rr] = 0
print("chosen: ", chosen)
print("index: ", rr)

nx.draw_networkx(G, pos, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()

node_list = list(G.nodes())
num_nodes = G.number_of_nodes()
color_map = np.full(num_nodes, 'blue')
labels= "S"
nx.set_node_attributes(G, labels, "labels")
lb = nx.get_node_attributes(G, 'labels')

nx.draw(G, pos, alpha=1, width=0.3, labels=lb, node_size=3000, node_color=color_map, font_size=36)
plt.show()

color_map[rr] = 'red'
nx.set_node_attributes(G, {chosen:{'labels':"I"}})
lb = nx.get_node_attributes(G, 'labels')

nx.draw(G, pos, alpha=1, width=0.3, labels=lb, node_size=3000, node_color=color_map, font_size=36)
plt.show()

d = 3
q = 0.7
p = 0.8

#loop
while(check_continue(color_map)):
    timelist = update_timelist(timelist)
    lb = check_recovery(color_map, G, node_list, timelist, d, q, lb)
    lb = check_spreading(color_map, G, node_list, p, lb)
    nx.draw(G, pos, alpha=1, width=0.3, labels=lb, node_size=3000, node_color=color_map, font_size=36)
    plt.show()