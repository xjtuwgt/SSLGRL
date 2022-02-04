import torch
from ogb.nodeproppred import DglNodePropPredDataset
from core.graph_utils import add_relation_ids_to_graph, construct_special_graph_dictionary
from torch import nn
from torch.utils.data import DataLoader
from core.gnn_layers import small_init_gain
from evens import HOME_DATA_FOLDER as ogb_root
from graph_data.graph_pretrained_dataloader import NodeSubGraphPairDataset
import logging
from utils.basic_utils import IGNORE_IDX


def ogb_nodeprop_graph_reconstruction(dataset: str):
    """
    :param dataset:
    'undirected':
       'ogbn-products': an undirected and unweighted graph (Amazon product)
       'ogbn-proteins': dataset is an undirected, weighted, and typed (according to species) graph
    'directed':
       'ogbn-arxiv': is a directed graph, representing the citation network (MAG) - Microsoft Academic Graph
       'ogbn-papers100M': dataset is a directed citation graph (MAG)
    'heterogeneous':
       'ogbn-mag' dataset is a heterogeneous network composed of a subset of the Microsoft Academic Graph
       directed relations
    :return:
    """
    data = DglNodePropPredDataset(name=dataset, root=ogb_root)
    node_split_idx = data.get_idx_split()
    graph, labels = data[0]
    # +++++++++++++++++++++++++++++++++
    graph.ndata['label'] = labels
    # +++++++++++++++++++++++++++++++++
    n_classes = labels.max().data.item()
    node_features = graph.ndata.pop('feat')
    n_feats = node_features.shape[1]
    if dataset in {'ogbn-products'}:  # 'ogbn-proteins'
        number_of_edges = graph.number_of_edges()
        edge_type_ids = torch.zeros(number_of_edges, dtype=torch.long)
        graph = add_relation_ids_to_graph(graph=graph, edge_type_ids=edge_type_ids)
        nentities, nrelations = graph.number_of_nodes(), 1
    elif dataset in {'ogbn-arxiv', 'ogbn-papers100M'}:
        number_of_edges = graph.number_of_edges()
        edge_type_ids = torch.zeros(number_of_edges, dtype=torch.long)
        graph = add_relation_ids_to_graph(graph=graph, edge_type_ids=edge_type_ids)
        src_nodes, dst_nodes = graph.edges()
        graph.add_edges(dst_nodes, src_nodes, {'rid': edge_type_ids + 1})
        nentities, nrelations = graph.number_of_nodes(), 2
    else:
        raise 'Dataset {} is not supported'.format(dataset)
    return graph, node_split_idx, node_features, nentities, nrelations, n_classes, n_feats


def ogb_k_hop_graph_reconstruction(dataset: str, hop_num=5, OON='zero'):
    logging.info('Bi-directional homogeneous graph: {}'.format(dataset))
    graph, node_split_idx, node_features, nentities, nrelations, n_classes, \
    n_feats = ogb_nodeprop_graph_reconstruction(dataset=dataset)
    graph, number_of_nodes, number_of_relations, \
    special_entity_dict, special_relation_dict = construct_special_graph_dictionary(graph=graph, n_entities=nentities,
                                                                                    n_relations=nrelations,
                                                                                    hop_num=hop_num)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    graph.ndata['label'][-2:] = IGNORE_IDX
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    number_of_added_nodes = number_of_nodes - nentities
    logging.info('Added number of nodes = {}'.format(number_of_added_nodes))
    assert len(special_entity_dict) == number_of_added_nodes
    if number_of_added_nodes > 0:
        added_node_features = torch.zeros((number_of_added_nodes, node_features.shape[1]), dtype=torch.float)
        if OON != 'zero':
            initial_weight = small_init_gain(d_in=node_features.shape[1], d_out=node_features.shape[1])
            added_node_features = nn.init.xavier_normal_(added_node_features.data.unsqueeze(0), gain=initial_weight)
            added_node_features = added_node_features.squeeze(0)
        node_features = torch.cat([node_features, added_node_features], dim=0)
    graph.ndata.update({'nid': torch.arange(0, number_of_nodes, dtype=torch.long)})
    return graph, node_split_idx, node_features, number_of_nodes, number_of_relations, \
           special_entity_dict, special_relation_dict, n_classes, n_feats


def ogb_train_valid_test(node_split_idx: dict, data_type: str):
    data_node_ids = node_split_idx[data_type]
    data_len = data_node_ids.shape[0]
    return data_len, data_node_ids


def ogb_subgraph_pretrain_dataloader(args):
    graph, node_split_idx, node_features, number_of_nodes, number_of_relations, special_entity_dict,\
    special_relation_dict, n_classes, n_feats = \
        ogb_k_hop_graph_reconstruction(dataset=args.ogb_node_name, hop_num=args.sub_graph_hop_num)
    logging.info('Number of nodes = {}'.format(number_of_nodes))
    args.node_number = number_of_nodes
    logging.info('Node features = {}'.format(n_feats))
    args.node_emb_dim = n_feats
    logging.info('Number of relations = {}'.format(number_of_relations))
    args.relation_number = number_of_relations
    logging.info('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
    fanouts = [int(_) for _ in args.sub_graph_fanouts.split(',')]
    ogb_dataset = NodeSubGraphPairDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
                                      special_entity2id=special_entity_dict,
                                      special_relation2id=special_relation_dict, fanouts=fanouts)
    ogb_dataloader = DataLoader(dataset=ogb_dataset, batch_size=args.per_gpu_pretrain_batch_size,
                                shuffle=True, pin_memory=True, drop_last=True,
                                collate_fn=NodeSubGraphPairDataset.collate_fn)
    return ogb_dataloader, node_features, n_classes

