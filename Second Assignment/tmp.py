import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

def check_continue(color_map):
    if 'red' in color_map:
        return True
    return False

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
print(list(G.nodes()))
rr = random.randrange(5)
chosen = list(G.nodes)[rr]
print("chosen: ", chosen)
print("index: ", rr)

nx.draw_networkx(G, pos, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()

color_map = []
for node in G: color_map.append('blue')

num_nodes = G.number_of_nodes()
labels=[]
nx.set_node_attributes(G, labels, "labels")
labels.append("S")
lb = nx.get_node_attributes(G, 'labels')
    
nx.draw(G, pos, alpha=1, width=0.3, labels=lb, node_size=3000, node_color=color_map, font_size=36)
plt.show()

color_map[rr] = 'red'
nx.set_node_attributes(G, {chosen:{'labels':"I"}})
lb = nx.get_node_attributes(G, 'labels')

nx.draw(G, pos, alpha=1, width=0.3, labels=lb, node_size=3000, node_color=color_map, font_size=36)
plt.show()

timelist = np.full(5, None)
print(timelist)

def update_timelist(timelist):
    #for all not None values -> +1
    return

#returns red nodes indexes
def get_I_nodes(color_map):
    lst = []
    x = 0
    for node in color_map:
        if node == 'red':
            lst.append(x)
        x+=1
    return lst

def check_recovery(color_map, G, d, q):
    #get all nodes with timelist > d
    #for each marked node: flip coin with probability q
    #if q -> color_map to green, label to R
    get_I_nodes(color_map)
    return

def check_spreading(color_map, G, p):
    #get all red nodes neighbors
    #for each marked node: flip coin with probability p
    #if p -> color_map to red, label to I
    I_nodes = get_I_nodes(color_map)
    print("red nodes indexes: ", I_nodes)
    for index in I_nodes:
        node = list(G.nodes)[index]
        neighbors = [n for n in G.neighbors(node)]
        print(neighbors)
        for n in neighbors:
            r = random.uniform(0.0, 1.0)
            if r <= p:
                color_map[n] = 'red' #can go out of bound, TO FIX
                nx.set_node_attributes(G, {list(G.nodes)[n]:{'labels':"I"}})

#TODO: create dictionary of node - color_map_index

d = 4
q = 0.3
p = 0.4

#loop
while(check_continue(color_map)):
    update_timelist(timelist)
    check_recovery(color_map, G, d, q)
    check_spreading(color_map, G, p)
    lb = nx.get_node_attributes(G, 'labels')
    nx.draw(G, pos, alpha=1, width=0.3, labels=lb, node_size=3000, node_color=color_map, font_size=36)
    plt.show()