import dgl
import numpy as np
import torch
from numpy import random
from dgl.sampling import sample_neighbors
from dgl.sampling.randomwalks import random_walk
from torch import Tensor
from time import time
from dgl import DGLHeteroGraph
import math
import copy


def construct_special_graph_dictionary(graph: DGLHeteroGraph, hop_num: int, n_relations: int, n_entities: int):
    """
    Add cls node to graph (last node), and extend multi-hop relation and cls relation and self-loop relation
    :param graph:
    :param hop_num: number of hops to generate special relations
    :param n_relations: number of relations in graph
    :param n_entities: number of entities (nodes) in graph
    :return:
    """
    special_entity_dict = {}
    special_relation_dict = {}
    number_nodes = n_entities
    assert number_nodes == graph.number_of_nodes()
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    special_entity_dict['cls'] = number_nodes  # for graph-level representation learning
    graph.add_nodes(1)  # add 'cls' node
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for hop in range(hop_num):
        special_relation_dict['in_hop_{}_r'.format(hop + 1)] = n_relations + (2 * hop)
        special_relation_dict['out_hop_{}_r'.format(hop + 1)] = n_relations + (2 * hop + 1)
    n_relations = n_relations + 2 * hop_num
    special_relation_dict['cls_r'] = n_relations  # connect each node to cls token;
    n_relations = n_relations + 1
    special_relation_dict['loop_r'] = n_relations  # self-loop relation
    n_relations = n_relations + 1
    number_of_nodes = graph.number_of_nodes()
    number_of_relations = n_relations
    return graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict


def add_relation_ids_to_graph(graph, edge_type_ids: Tensor):
    """
    Add relation ids on edges
    :param graph:
    :param edge_type_ids: add 'rid' to graph edge data --> type id
    :return:
    """
    graph.edata['rid'] = edge_type_ids
    return graph


def sub_graph_neighbor_sample(graph: DGLHeteroGraph, anchor_node_ids: Tensor, cls_node_ids: Tensor,
                              fanouts: list, edge_dir: str = 'in', unique_neighbor: bool = False,
                              debug=False):
    """
    Neighbor-hood based sub-graph sampling
    :param unique_neighbor:
    :param graph: dgl graph
    :param anchor_node_ids: LongTensor
    :param cls_node_ids: LongTensor
    :param fanouts: size = hop_number, (list, each element represents the number of sampling neighbors)
    :param edge_dir:  'in' or 'out'
    :param debug:
    :return:
    """
    assert edge_dir in {'in', 'out'}
    start_time = time() if debug else 0
    neighbors_dict = {'anchor': anchor_node_ids, 'cls': cls_node_ids}
    edge_dict = {}  # sampled edge dictionary: (head, t_id, tail)
    hop, hop_number = 1, len(fanouts)
    while hop < hop_number + 1:
        if hop == 1:
            node_ids = neighbors_dict['anchor']
        else:
            node_ids = neighbors_dict['{}_hop_{}'.format(edge_dir, hop - 1)]
        sg = sample_neighbors(g=graph, nodes=node_ids, edge_dir=edge_dir, fanout=fanouts[hop - 1])
        sg_src, sg_dst = sg.edges()
        sg_eids, sg_tids = sg.edata[dgl.EID], sg.edata['rid']
        sg_src_list, sg_dst_list = sg_src.tolist(), sg_dst.tolist()
        sg_eid_list, sg_tid_list = sg_eids.tolist(), sg_tids.tolist()
        for _, eid in enumerate(sg_eid_list):
            edge_dict[eid] = (sg_src_list[_], sg_tid_list[_], sg_dst_list[_])
        hop_neighbor = sg_src if edge_dir == 'in' else sg_dst
        if unique_neighbor:
            hop_neighbor = torch.unique(hop_neighbor)
        neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = hop_neighbor
        hop = hop + 1
    end_time = time() if debug else 0
    if debug:
        print('Sampling time = {:.4f} seconds'.format(end_time - start_time))
    neighbors_dict = dict([(k, torch.unique(v, return_counts=True)) for k, v in neighbors_dict.items()])
    # #############################################################################################
    node_arw_label_dict = {anchor_node_ids[0].data.item(): 1, cls_node_ids[0].data.item(): 0}
    # ###########################################anonymous rand walk node labels###################
    for hop in range(1, hop_number + 1):
        hop_neighbors = neighbors_dict['{}_hop_{}'.format(edge_dir, hop)]
        for neighbor in hop_neighbors[0].tolist():
            if neighbor not in node_arw_label_dict:
                node_arw_label_dict[neighbor] = hop + 1
    ##############################################################################################
    return neighbors_dict, node_arw_label_dict, edge_dict


def sub_graph_rwr_sample(graph: DGLHeteroGraph, anchor_node_ids: Tensor, cls_node_ids: Tensor,
                         fanouts: list, restart_prob: float = 0.8, edge_dir: str = 'in', debug=False):
    """
    Random walk with re-strart based sub-graph sampling
    :param restart_prob:
    :param graph: graph have edge type: rid
    :param anchor_node_ids:
    :param cls_node_ids:
    :param fanouts:
    :param edge_dir:
    :param debug:
    :return:
    """
    assert edge_dir in {'in', 'out'}
    start_time = time() if debug else 0
    if edge_dir == 'in':
        raw_graph = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
    else:
        raw_graph = graph
    # ##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    walk_length = len(fanouts)
    num_traces = max(64,
                     int((graph.out_degrees(anchor_node_ids.data.item()) * math.e
                          / (math.e - 1) / restart_prob) + 0.5),
                     torch.prod(torch.tensor(fanouts, dtype=torch.long)).data.item())  # Number of sampled traces
    num_traces = num_traces * 5
    assert num_traces > 1
    neighbors_dict = {'anchor': (anchor_node_ids, torch.tensor([1], dtype=torch.long)),
                      'cls': (cls_node_ids, torch.tensor([1], dtype=torch.long))}
    node_pos_label_dict = {anchor_node_ids[0].data.item(): 1, cls_node_ids[0].data.item(): 0}
    edge_dict = {}  # sampled edge dictionary: (head, t_id, tail)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    anchor_node_ids = anchor_node_ids.repeat(num_traces)
    traces, _ = random_walk(g=raw_graph, nodes=anchor_node_ids, length=walk_length, restart_prob=restart_prob)
    valid_trace_idx = (traces >= 0).sum(dim=1) > 1
    traces = traces[valid_trace_idx]
    for hop in range(1, walk_length + 1):
        trace_i = traces[:, hop]
        trace_i = trace_i[trace_i >= 0]
        if trace_i.shape[0] > 0:
            hop_neighbors = torch.unique(trace_i, return_counts=True)
            neighbors_dict['{}_hop_{}'.format(edge_dir, hop)] = hop_neighbors
            for neighbor in hop_neighbors[0].tolist():
                if neighbor not in node_pos_label_dict:
                    node_pos_label_dict[neighbor] = hop + 1
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    src_nodes, dst_nodes = traces[:, :-1].flatten(), traces[:, 1:].flatten()
    valid_edge_idx = dst_nodes >= 0
    src_nodes, dst_nodes = src_nodes[valid_edge_idx], dst_nodes[valid_edge_idx]
    if edge_dir == 'in':
        edge_ids = graph.edge_ids(dst_nodes, src_nodes)
    else:
        edge_ids = graph.edge_ids(src_nodes, dst_nodes)
    edge_tids = graph.edata['rid'][edge_ids]
    eid_list, tid_list = edge_ids.tolist(), edge_tids.tolist()
    src_node_list, dst_node_list = src_nodes.tolist(), dst_nodes.tolist()
    for _, eid in enumerate(eid_list):
        edge_dict[eid] = (src_node_list[_], tid_list[_], dst_node_list[_])
    ##############################################################################################
    end_time = time() if debug else 0
    if debug:
        print('RWR Sampling time = {:.4f} seconds'.format(end_time - start_time))
    return neighbors_dict, node_pos_label_dict, edge_dict


def sub_graph_construction(graph, edge_dict: dict, neighbors_dict: dict, bi_directed: bool = True):
    """
    Construct sub-graph based on edge_dict (based on the sampled edges)
    :param graph: original graph
    :param edge_dict: edge dictionary: eid--> (src_node, edge_type, dst_node)
    :param neighbors_dict: {cls, anchor, hop} -> ((neighbors, neighbor counts))
    :param bi_directed: whether get bi-directional graph
    :return:
    """
    if len(edge_dict) == 0:
        assert 'anchor' in neighbors_dict
        return single_node_graph_extractor(graph=graph, neighbors_dict=neighbors_dict)
    edge_ids = list(edge_dict.keys())
    if bi_directed:  # the graph is bi-directed
        parent_triples = np.array(list(edge_dict.values()))
        rev_edge_ids = graph.edge_ids(parent_triples[:, 2], parent_triples[:, 0]).tolist()
        rev_edge_ids = [_ for _ in rev_edge_ids if _ not in edge_dict]  # adding new edges as graph is bi_directed
        rev_edge_ids = sorted(set(rev_edge_ids), key=rev_edge_ids.index)
    else:
        rev_edge_ids = []
    edge_ids = edge_ids + rev_edge_ids
    subgraph = graph.edge_subgraph(edges=edge_ids)
    return subgraph


def single_node_graph_extractor(graph, neighbors_dict: dict):
    """
    One isolate node based sub-graph extraction
    :param graph:
    :param neighbors_dict: int --> (anchor_ids, anchor_counts)
    :return:
    """
    anchor_ids = neighbors_dict['anchor'][0]
    sub_graph = graph.subgraph(anchor_ids)
    return sub_graph


def add_self_loop_to_graph(graph, self_loop_r: int):
    """
    Add self-loop relation to the graph
    :param graph:
    :param self_loop_r:
    :return:
    """
    g = copy.deepcopy(graph)
    number_of_nodes = g.number_of_nodes()
    self_loop_r_array = torch.full((number_of_nodes,), self_loop_r, dtype=torch.long)
    node_ids = torch.arange(number_of_nodes)
    g.add_edges(node_ids, node_ids, {'rid': self_loop_r_array})
    return g


def cls_node_addition_to_graph(subgraph, cls_parent_node_id: int, special_relation_dict: dict):
    """
    Add one cls node into sub-graph as super-node, and connect the cls node with other nodes with 'cls_r' relation type
    :param subgraph:
    :param cls_parent_node_id: cls node shared across all subgraphs
    :param special_relation_dict: {cls_r: cls_r index}
    :return: sub_graph added cls_node (for graph level representation learning
    """
    assert 'cls_r' in special_relation_dict
    subgraph.add_nodes(1)  # the last node is the cls_node
    subgraph.ndata['nid'][-1] = cls_parent_node_id  # set the nid (parent node id) in sub-graph
    parent_node_ids, sub_node_ids = subgraph.ndata['nid'].tolist(), subgraph.nodes().tolist()
    parent2sub_dict = dict(zip(parent_node_ids, sub_node_ids))
    cls_idx = parent2sub_dict[cls_parent_node_id]
    assert cls_idx == subgraph.number_of_nodes() - 1
    cls_relation = [special_relation_dict['cls_r']] * (2 * (subgraph.number_of_nodes() - 1))
    cls_relation = torch.tensor(cls_relation, dtype=torch.long)
    cls_src_nodes = [cls_idx] * (subgraph.number_of_nodes() - 1)
    cls_src_nodes = torch.tensor(cls_src_nodes, dtype=torch.long)
    cls_dst_nodes = torch.arange(0, subgraph.number_of_nodes() - 1)  # the last node is the cls node
    cls_src, cls_dst = torch.cat((cls_src_nodes, cls_dst_nodes)), np.concatenate((cls_dst_nodes, cls_src_nodes))
    # bi-directional cls_nodes
    subgraph.add_edges(cls_src, cls_dst, {'rid': cls_relation})
    return subgraph, parent2sub_dict


def anchor_node_sub_graph_extractor(graph, anchor_node_ids: Tensor, cls_node_ids: Tensor, fanouts: list,
                                    special_relation2id: dict, samp_type: str = 'ns', restart_prob: float = 0.8,
                                    edge_dir: str = 'in', self_loop: bool = True, bi_directed: bool = True,
                                    cls_addition: bool = True, ns_unique_neighbor: bool = False, debug=False):
    """
    Extract Sub-graph from graph based o given anchor point
    :param graph:
    :param anchor_node_ids:
    :param cls_node_ids:
    :param fanouts:
    :param special_relation2id:
    :param samp_type:
    :param restart_prob:
    :param edge_dir:
    :param self_loop:
    :param bi_directed:
    :param cls_addition:
    :param ns_unique_neighbor:
    :param debug:
    :return:
    """
    if samp_type == 'ns':
        neighbors_dict, node_pos_label_dict, edge_dict = sub_graph_neighbor_sample(graph=graph,
                                                                                   anchor_node_ids=anchor_node_ids,
                                                                                   cls_node_ids=cls_node_ids,
                                                                                   fanouts=fanouts,
                                                                                   edge_dir=edge_dir,
                                                                                   unique_neighbor=ns_unique_neighbor,
                                                                                   debug=debug)
    elif samp_type == 'rwr':
        neighbors_dict, node_pos_label_dict, edge_dict = sub_graph_rwr_sample(graph=graph,
                                                                              anchor_node_ids=anchor_node_ids,
                                                                              cls_node_ids=cls_node_ids,
                                                                              fanouts=fanouts,
                                                                              restart_prob=restart_prob,
                                                                              edge_dir=edge_dir,
                                                                              debug=debug)
    else:
        raise 'Sampling method {} is not supported!'.format(samp_type)
    subgraph = sub_graph_construction(graph=graph, edge_dict=edge_dict, bi_directed=bi_directed,
                                      neighbors_dict=neighbors_dict)

    if cls_addition:
        cls_parent_node_id = neighbors_dict['cls'][0][0].data.item()
        subgraph, parent2sub_dict = cls_node_addition_to_graph(subgraph=subgraph,
                                                               special_relation_dict=special_relation2id,
                                                               cls_parent_node_id=cls_parent_node_id)
    else:
        parent_node_ids, sub_node_ids = subgraph.ndata['nid'].tolist(), subgraph.nodes().tolist()
        parent2sub_dict = dict(zip(parent_node_ids, sub_node_ids))

    if self_loop:
        subgraph = add_self_loop_to_graph(graph=subgraph, self_loop_r=special_relation2id['loop_r'])
    assert len(parent2sub_dict) == subgraph.number_of_nodes()
    node_orders = torch.zeros(len(parent2sub_dict), dtype=torch.long).to(graph.device)
    for key, value in parent2sub_dict.items():
        node_orders[value] = node_pos_label_dict[key]
    subgraph.ndata['n_rw_pos'] = node_orders
    return subgraph, parent2sub_dict, neighbors_dict


"""
Graph augmentation methods based on edge perturbations (adding new edges and removing edges)
1) Adding edges
   a) add edges among multi-hop neighbors
   b) add edges between anchor-point and multi-hop neighbors
   c) add k-hop edges
2) Removing edges (constraint: removing edges will not change the graph connectivity)
   a) removing the edges among multi-hop neighbors
"""


def self_loop_augmentation(subgraph, self_loop_r: int):
    aug_sub_graph = copy.deepcopy(subgraph)
    number_of_nodes = subgraph.number_of_nodes()
    node_ids = torch.arange(number_of_nodes - 1)
    self_loop_r = torch.full((number_of_nodes - 1,), self_loop_r, dtype=torch.long)
    aug_sub_graph.add_edges(node_ids, node_ids, {'rid': self_loop_r})
    assert subgraph.number_of_nodes() == aug_sub_graph.number_of_nodes()
    return aug_sub_graph


def graph_augmentation_via_edge_addition(subgraph):
    return
