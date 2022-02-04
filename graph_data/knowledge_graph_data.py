from evens import KG_DATA_FOLDER
from utils.kg_utils import KGDataset, kg_data_path_collection, \
    knowledge_graph_construction_from_triples
from utils.graph_utils import construct_special_graph_dictionary
from graph_data.graph_pretrained_dataloader import NodeSubGraphPairDataset
from torch.utils.data import DataLoader
import logging


def knowledge_graph_khop_reconstruction(dataset: str, hop_num=5, bi_directed: bool = True):
    print('Bi-directional homogeneous graph: {}'.format(dataset))
    entity_path, relation_path, train_path, _, _ = kg_data_path_collection(kg_path=KG_DATA_FOLDER, kg_name=dataset)
    kg_data = KGDataset(entity_path=entity_path, relation_path=relation_path, train_path=train_path)
    graph = knowledge_graph_construction_from_triples(num_entities=kg_data.n_entities,
                                                      num_relations=kg_data.n_relations,
                                                      triples=kg_data.train, bi_directional=bi_directed)
    nentities = kg_data.n_entities
    nrelations = 2 * kg_data.n_relations if bi_directed else kg_data.n_relations
    graph, number_of_nodes, number_of_relations, \
    special_entity_dict, special_relation_dict = construct_special_graph_dictionary(graph=graph, n_entities=nentities,
                                                                                    n_relations=nrelations,
                                                                                    hop_num=hop_num)
    number_of_added_nodes = number_of_nodes - nentities
    print('Added number of nodes = {}'.format(number_of_added_nodes))
    assert len(special_entity_dict) == number_of_added_nodes
    print('Graph nodes: {}, edges: {}'.format(graph.number_of_nodes(), graph.number_of_edges()))
    return graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict


def kg_subgraph_pretrain_dataloader(args):
    graph, number_of_nodes, number_of_relations, special_entity_dict, special_relation_dict = \
        knowledge_graph_khop_reconstruction(dataset=args.kg_name, hop_num=args.sub_graph_hop_num)
    logging.info('Number of nodes = {}'.format(number_of_nodes))
    args.node_number = number_of_nodes
    logging.info('Number of relations = {}'.format(number_of_relations))
    args.relation_number = number_of_relations
    logging.info('Number of nodes with 0 in-degree = {}'.format((graph.in_degrees() == 0).sum()))
    fanouts = [int(_) for _ in args.sub_graph_fanouts.split(',')]
    kg_dataset = NodeSubGraphPairDataset(graph=graph, nentity=number_of_nodes, nrelation=number_of_relations,
                                         special_entity2id=special_entity_dict,
                                         special_relation2id=special_relation_dict, fanouts=fanouts)
    kg_dataloader = DataLoader(dataset=kg_dataset, batch_size=args.per_gpu_train_batch_size,
                               shuffle=True, pin_memory=True, drop_last=True,
                               collate_fn=NodeSubGraphPairDataset.collate_fn)
    return kg_dataloader
