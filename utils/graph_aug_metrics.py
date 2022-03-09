from dgl import DGLHeteroGraph
import numpy as np
import dgl


def degree_distribution(graph: DGLHeteroGraph, edge_dir: str = 'out'):
    g = dgl.remove_self_loop(graph)
    if edge_dir == 'out':
        degrees = g.out_degrees().cpu().detach().numpy().astype(np.float)
    else:
        degrees = g.in_degrees().cpu().detach().numpy().astype(np.float)
    return degrees


def local_cluster_coefficients(graph: DGLHeteroGraph):
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


def rbf_kernel_func(graph_1_vec, graph_2_vec, num_bins: int):

    return


def Max_Mean_Dist(g1: DGLHeteroGraph, g2: DGLHeteroGraph):

    return

