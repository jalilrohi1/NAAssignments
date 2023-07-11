import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import logging

G = nx.fast_gnp_random_graph(100, 0.1, 420)
pos = nx.spring_layout(G, seed=420)
nx.draw_networkx(G, pos)
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()

dg = sorted(G.degree, key=lambda x: x[1], reverse=True)
print(min(dg))

H =  nx.barabasi_albert_graph(100, 2, 420)
pos = nx.spring_layout(H, seed=420)
nx.draw_networkx(H, pos)
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()

dg = sorted(H.degree, key=lambda x: x[1], reverse=True)
print(min(dg))