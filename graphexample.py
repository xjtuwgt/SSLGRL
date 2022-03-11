import networkx as nx
from itertools import combinations
from random import random
import dgl
import numpy as np
import torch
from dgl.sampling import sample_neighbors

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


# n = 4
# p = 0.65
# G = ER(n, p)
# print(type(G))
#
# # graph = dgl.from_networkx(nx_graph=G)
# graph = dgl.rand_graph(4, 8)
# print(graph)
# adj_matrix = graph.adj()
# print(adj_matrix.to_dense())
# A = adj_matrix.to_dense()
# D = torch.diag(1.0/A.sum(dim=1))
# A_n = torch.matmul(D, A)
# print(A_n)
# print(A_n.sum(dim=1))
#
# A2 = torch.matmul(A_n, A_n)
# print(A2.sum(dim=1))
# A3 = torch.matmul(A2, A_n)
# A4 = torch.matmul(A3, A_n)
# A5 = torch.matmul(A4, A_n)
# A6 = torch.matmul(A5, A_n)
# print(A6)

src_nodes = [0, 0, 0]
dst_nodes = [1, 2, 3]

graph = dgl.graph((src_nodes + dst_nodes, dst_nodes +  src_nodes))
# print(graph)
graph.edata['rid'] = torch.zeros(graph.number_of_edges())
# print(graph.nodes())
from utils.graph_aug_metrics import local_cluster_coefficients, degree_distribution

# print(graph.out_degrees().cpu().detach().numpy())

# x = graph.adj(scipy_fmt='coo').toarray()
# print(type(x))
# print(x)

y = local_cluster_coefficients(graph=graph)
print(y)
y = degree_distribution(graph=graph)
print(y)

graph.add_edges([1,2], [2,1])

y = local_cluster_coefficients(graph=graph)
print(y)
y = degree_distribution(graph=graph)
print(y)


#
# from utils.graph_utils import sub_graph_neighbor_sample
#
# neighbors_dict, node_arw_label_dict, edge_dict = \
#     sub_graph_neighbor_sample(graph=graph, anchor_node_ids=torch.LongTensor([0]),
#                           cls_node_ids=torch.LongTensor([7]), fanouts=[-1])
#
# print(neighbors_dict)
# print(edge_dict)
# print(node_arw_label_dict)

# src_nodes = [0, 0, 0, 1]
# dst_nodes = [1, 2, 3, 2]
# graph = dgl.graph((src_nodes + dst_nodes, dst_nodes + src_nodes))
# graph.edata['rid'] = torch.zeros(graph.number_of_edges(), dtype=torch.long)
#
# neighbors_dict, node_arw_label_dict, edge_dict = \
#     sub_graph_neighbor_sample(graph=graph, anchor_node_ids=torch.LongTensor([0]),
#                               cls_node_ids=torch.LongTensor([4]), fanouts=[-1, -1])
# print(neighbors_dict)
# print(node_arw_label_dict)
# print(edge_dict)
# z = np.array([[2,2,2]] * 3)
# z[1][0] = 1
# z[1][1] = 1
# z[1][2] = 1
# x = np.array([[1, 0.8, 0], [0.8, 1, 0.8], [0, 0.8, 1.0]])
# x = x / x.sum(axis=0)[:,None]
# print(np.matmul(x, z))
# # print(x)
# norm_x = np.eye(3) - 0.3 * x
# # print(norm_x)
# inv_x = np.linalg.inv(norm_x)
# # print(inv_x)
# print(np.matmul(inv_x, z))
# # print(x)
# # x2 = np.matmul(x,x)
# # print(x2)
# # x3 = np.matmul(x2, x)
# # print(x3)
# # x4 = np.matmul(x3, x)
# # print(x4)
# # z = np.ones((4,3))
# z = np.array([[2,2,2]] * 4)
# z[1][0] = 1
# z[1][1] = 1
# z[1][2] = 1
# x = np.array([[1,0.8,0,0], [0.8, 1, 0.8, 0.8], [0, 0.8, 1, 0], [0, 0.8, 0, 1]])
# x = x / x.sum(axis=0)[:,None]
# print(np.matmul(x, z))
# # print(x)
# norm_x = np.eye(4) - 0.3 * x
# # print(norm_x)
# inv_x = np.linalg.inv(norm_x)
# # print(inv_x)
# print(np.matmul(inv_x, z))