from dgl import DGLHeteroGraph
import numpy as np
from numpy import ndarray
import dgl


def degree_distribution(graph: DGLHeteroGraph, edge_dir: str = 'out') -> ndarray:
    g = dgl.remove_self_loop(graph)
    if edge_dir == 'out':
        degrees = g.out_degrees().cpu().detach().numpy().astype(np.float)
    else:
        degrees = g.in_degrees().cpu().detach().numpy().astype(np.float)
    return degrees


def local_cluster_coefficients(graph: DGLHeteroGraph) -> ndarray:
    adj_matrix = graph.adj(scipy_fmt='coo').toarray()
    adj_matrix[adj_matrix > 1] = 1  # delete the multiple edges if there are multiple edges among nodes
    np.fill_diagonal(adj_matrix, 0)
    node_num = adj_matrix.shape[0]
    cluster_coeff_array = np.zeros(node_num)
    for _ in range(node_num):
        row = adj_matrix[_]
        row_idx = np.where(row == 1)[0]
        neighbor_num = len(row_idx)
        if neighbor_num > 1:
            neighbor_adj_matrix = adj_matrix[np.ix_(row_idx, row_idx)]
            neighbor_connections = neighbor_adj_matrix.sum() * 1.0
            local_cluster_coeff = neighbor_connections / (neighbor_num * (neighbor_num - 1))
            assert local_cluster_coeff <= 1.0
            cluster_coeff_array[_] = local_cluster_coeff
    return cluster_coeff_array


def graph_vectorization(graph: DGLHeteroGraph, vector_func='degree') -> ndarray:
    assert vector_func in {'degree', 'cluster_coefficient'}
    if vector_func == 'degree':
        graph_vector = degree_distribution(graph=graph)
    else:
        graph_vector = local_cluster_coefficients(graph=graph)
    return graph_vector


def distribution_vector(graph_vector: ndarray, n_bins: int):
    return None


def rbf_kernel_func(graph_1_vec: ndarray, graph_2_vec: ndarray, delta: float = 0.1):

    return


def Max_Mean_Dist(g1: DGLHeteroGraph, g2: DGLHeteroGraph, delta: float, n_bins: int = 100, vector_func='degree'):
    g1_vector = graph_vectorization(graph=g1, vector_func=vector_func)
    g2_vector = graph_vectorization(graph=g2, vector_func=vector_func)
    assert g1_vector.shape[0] == g2_vector.shape[0]
    if vector_func == 'degree':
        max_degree = max([g1_vector.max(), g2_vector.max()])
        vector_n_bin = int(max_degree)
    else:
        vector_n_bin = min(n_bins, g1_vector.shape[0])


    return

