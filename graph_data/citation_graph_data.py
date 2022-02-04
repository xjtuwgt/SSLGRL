import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from graph_data.graph_pretrained_dataset import NodeSubGraphPairDataset
from graph_data.graph_train_dataset import node_prediction_data_helper
from torch.utils.data import DataLoader
from core.graph_utils import add_relation_ids_to_graph, construct_special_graph_dictionary
import torch.nn.init as INIT
from utils.basic_utils import IGNORE_IDX
from core.gnn_layers import small_init_gain
import logging


def citation_graph_reconstruction(dataset: str):
    if dataset == 'cora':
        data = CoraGraphDataset()
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    graph = data[0]
    n_classes = data.num_labels
    node_features = graph.ndata.pop('feat')
    n_feats = node_features.shape[1]
    number_of_edges = graph.number_of_edges()
    edge_type_ids = torch.zeros(number_of_edges, dtype=torch.long)
    graph = add_relation_ids_to_graph(graph=graph, edge_type_ids=edge_type_ids)
    nentities, nrelations = graph.number_of_nodes(), 1
    return graph, node_features, nentities, nrelations, n_classes, n_feats


def citation_khop_graph_reconstruction(dataset: str, hop_num=5, OON='zero'):
    logging.info('Bi-directional homogeneous graph: {}'.format(dataset))
    graph, node_features, nentities, nrelations, n_classes, n_feats = citation_graph_reconstruction(dataset=dataset)
    graph, number_of_nodes, number_of_relations, \
    special_entity_dict, special_relation_dict = construct_special_graph_dictionary(graph=graph, n_entities=nentities,
                                                                                    n_relations=nrelations,
                                                                                    hop_num=hop_num)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    graph.ndata['label'][-2:] = -IGNORE_IDX
    graph.ndata['val_mask'][-2:] = False
    graph.ndata['train_mask'][-2:] = False
    graph.ndata['test_mask'][-2:] = False
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    number_of_added_nodes = number_of_nodes - nentities
    logging.info('Added number of nodes = {}'.format(number_of_added_nodes))
    assert len(special_entity_dict) == number_of_added_nodes
    if number_of_added_nodes > 0:
        added_node_features = torch.zeros((number_of_added_nodes, node_features.shape[1]), dtype=torch.float)
        if OON != 'zero':
            initial_weight = small_init_gain(d_in=node_features.shape[1], d_out=node_features.shape[1])
            added_node_features = INIT.xavier_normal_(added_node_features.data.unsqueeze(0), gain=initial_weight)
            added_node_features = added_node_features.squeeze(0)
        node_features = torch.cat([node_features, added_node_features], dim=0)
    graph.ndata.update({'nid': torch.arange(0, number_of_nodes, dtype=torch.long)})
    return graph, node_features, number_of_nodes, number_of_relations, \
           special_entity_dict, special_relation_dict, n_classes, n_feats


def citation_subgraph_pretrain_dataloader(args):
    graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, \
    special_relation_dict, n_classes, n_feats = \
        citation_khop_graph_reconstruction(dataset=args.citation_name, hop_num=args.sub_graph_hop_num)
    logging.info('Number of nodes = {}'.format(number_of_nodes))
    args.node_number = number_of_nodes
    logging.info('Node features = {}'.format(n_feats))
    args.node_emb_dim = n_feats
    logging.info('Number of relations = {}'.format(number_of_relations))
    args.relation_number = number_of_relations
    logging.info('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
    fanouts = [int(_) for _ in args.sub_graph_fanouts.split(',')]
    citation_dataset = NodeSubGraphPairDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
                                           special_entity2id=special_entity_dict,
                                           special_relation2id=special_relation_dict, fanouts=fanouts)
    citation_dataloader = DataLoader(dataset=citation_dataset, batch_size=args.per_gpu_pretrain_batch_size,
                                     shuffle=True, pin_memory=True, drop_last=True,
                                     collate_fn=NodeSubGraphPairDataset.collate_fn)
    return citation_dataloader, node_features, n_classes


def citation_node_pred_subgraph_data_helper(args):
    graph, node_features, number_of_nodes, number_of_relations, special_entity_dict, \
    special_relation_dict, n_classes, n_feats = \
        citation_khop_graph_reconstruction(dataset=args.citation_name, hop_num=args.sub_graph_hop_num)
    logging.info('Number of nodes = {}'.format(number_of_nodes))
    args.node_number = number_of_nodes
    logging.info('Node features = {}'.format(n_feats))
    args.node_emb_dim = n_feats
    logging.info('Number of relations = {}'.format(number_of_relations))
    args.relation_number = number_of_relations
    logging.info('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
    # fanouts = [int(_) for _ in args.sub_graph_fanouts.split(',')]
    fanouts = [-1 for _ in args.sub_graph_fanouts.split(',')]
    data_helper = node_prediction_data_helper(graph=graph, fanouts=fanouts, number_of_nodes=number_of_nodes,
                                              number_of_relations=number_of_relations,
                                              special_entity_dict=special_entity_dict,
                                              special_relation_dict=special_relation_dict,
                                              num_class=n_classes,
                                              self_loop=args.sub_graph_self_loop,
                                              train_batch_size=args.train_batch_size, node_split_idx=None,
                                              val_batch_size=args.eval_batch_size, graph_type=args.graph_type)
    return data_helper
