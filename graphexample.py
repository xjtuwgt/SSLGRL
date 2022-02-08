import networkx as nx
from itertools import combinations
from random import random
import dgl
import torch

from utils.basic_utils import seed_everything

seed_everything(seed=46)

def ER(n, p):
    V = set([v for v in range(n)])
    E = set()
    for combination in combinations(V, 2):
        a = random()
        if a < p:
            E.add(combination)

    g = nx.Graph()
    g.add_nodes_from(V)
    g.add_edges_from(E)

    return g


n = 4
p = 0.65
G = ER(n, p)
print(type(G))

# graph = dgl.from_networkx(nx_graph=G)
graph = dgl.rand_graph(4, 8)
print(graph)
adj_matrix = graph.adj()
print(adj_matrix.to_dense())
A = adj_matrix.to_dense()
D = torch.diag(1.0/A.sum(dim=1))
A_n = torch.matmul(D, A)
print(A_n)
print(A_n.sum(dim=1))

A2 = torch.matmul(A_n, A_n)
print(A2.sum(dim=1))
A3 = torch.matmul(A2, A_n)
A4 = torch.matmul(A3, A_n)
A5 = torch.matmul(A4, A_n)
A6 = torch.matmul(A5, A_n)
print(A6)