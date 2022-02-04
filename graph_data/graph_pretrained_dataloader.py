from dgl import DGLHeteroGraph
import torch
import dgl
from numpy import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from graph_data.citation_graph_data import citation_k_hop_graph_reconstruction
from graph_data.ogb_graph_data import ogb_k_hop_graph_reconstruction
from core.graph_utils import sub_graph_neighbor_sample, cls_sub_graph_extractor, \
    cls_anchor_sub_graph_augmentation


class NodeSubGraphPairDataset(Dataset):
    def __init__(self, graph: DGLHeteroGraph, nentity: int, nrelation: int, fanouts: list,
                 special_entity2id: dict, special_relation2id: dict, bi_directed=True, edge_dir='in'):
        assert len(fanouts) > 0
        self.fanouts = fanouts  # list of int == number of hops for sampling
        self.hop_num = len(fanouts)
        self.g = graph
        #####################
        if len(special_entity2id) > 0:
            self.len = graph.number_of_nodes() - len(special_entity2id)  # no sub-graph extraction on special entities
        else:
            self.len = graph.number_of_nodes()
        #####################
        self.nentity, self.nrelation = nentity, nrelation
        self.bi_directed = bi_directed
        self.edge_dir = edge_dir  # "in", "out"
        self.special_entity2id, self.special_relation2id = special_entity2id, special_relation2id

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        anchor_node_ids = torch.LongTensor([idx])
        samp_hop_num = random.randint(2, self.hop_num + 1)
        samp_fanouts = self.fanouts[:samp_hop_num]
        cls_node_ids = torch.LongTensor([self.special_entity2id['cls']])
        neighbors_dict, node_arw_label_dict, edge_dict = \
            sub_graph_neighbor_sample(graph=self.g, anchor_node_ids=anchor_node_ids,
                                      cls_node_ids=cls_node_ids, fanouts=samp_fanouts,
                                      edge_dir=self.edge_dir, debug=False)
        subgraph, parent2sub_dict = cls_sub_graph_extractor(graph=self.g, edge_dict=edge_dict,
                                                            neighbors_dict=neighbors_dict,
                                                            special_relation_dict=self.special_relation2id,
                                                            node_arw_label_dict=node_arw_label_dict,
                                                            bi_directed=self.bi_directed, debug=False)

        aug_subgraph = cls_anchor_sub_graph_augmentation(subgraph=subgraph, parent2sub_dict=parent2sub_dict,
                                                         neighbors_dict=neighbors_dict, edge_dir=self.edge_dir,
                                                         special_relation_dict=self.special_relation2id)

        sub_anchor_id = parent2sub_dict[idx.data.item()]
        assert subgraph.number_of_nodes() == aug_subgraph.number_of_nodes()
        return subgraph, aug_subgraph, sub_anchor_id

    @staticmethod
    def collate_fn(data):
        assert len(data[0]) == 3
        batch_graphs_1 = dgl.batch([_[0] for _ in data])
        batch_graphs_2 = dgl.batch([_[1] for _ in data])

        batch_graph_cls = torch.as_tensor([_[0].number_of_nodes() for _ in data], dtype=torch.long)
        batch_graph_cls = torch.cumsum(batch_graph_cls, dim=0) - 1
        # ++++++++++++++++++++++++++++++++++++++++
        batch_anchor_id = torch.zeros(len(data), dtype=torch.long)
        for idx, _ in enumerate(data):
            if idx == 0:
                batch_anchor_id[idx] = _[2]
            else:
                batch_anchor_id[idx] = _[2] + batch_graph_cls[idx - 1].data.item() + 1
        # +++++++++++++++++++++++++++++++++++++++
        return {'batch_graph_1': (batch_graphs_1, batch_graph_cls, batch_anchor_id),
                'batch_graph_2': (batch_graphs_2, batch_graph_cls, batch_anchor_id)}


class PretrainedGraphDataHelper(object):
    def __init__(self, config):
        self.config = config
        self.graph_type = self.config.graph_type
        assert self.graph_type in {'citation', 'ogb'}
        if self.graph_type == 'citation':
            graph, node_features, number_of_nodes, number_of_relations, \
            special_node_dict, special_relation_dict, n_classes, n_feats = \
                citation_k_hop_graph_reconstruction(dataset=self.config.citation_node_name,
                                                    hop_num=self.config.sub_graph_hop_num,
                                                    OON=self.config.oon_type)
            self.node_split_idx = None
        elif self.graph_type == 'ogb':
            graph, node_split_idx, node_features, number_of_nodes, number_of_relations, \
            special_node_dict, special_relation_dict, n_classes, n_feats = ogb_k_hop_graph_reconstruction(
                dataset=self.config.ogb_node_name,
                hop_num=self.config.sub_graph_hop_num,
                OON=self.config.oon_type)
            self.node_split_idx = node_split_idx
        else:
            raise '{} is not supported'.format(self.graph_type)
        self.graph = graph
        self.number_of_nodes = number_of_nodes
        self.number_of_relations = number_of_relations
        self.num_class = n_classes
        self.n_feats = n_feats
        self.node_features = node_features
        self.special_entity_dict = special_node_dict
        self.special_relation_dict = special_relation_dict
        self.train_batch_size = self.config.train_batch_size
        self.val_batch_size = self.config.eval_batch_size
        self.edge_dir = self.config.sub_graph_edge_dir
        self.self_loop = self.config.sub_graph_self_loop  # whether adding self-loop in sub-graph
        self.fanouts = [int(_) for _ in self.config.sub_graph_fanouts.split(',')]

    def data_loader(self):
        dataset = NodeSubGraphPairDataset(graph=self.graph,
                                          nentity=self.number_of_nodes,
                                          nrelation=self.number_of_relations,
                                          special_entity2id=self.special_entity_dict,
                                          special_relation2id=self.special_relation_dict,
                                          edge_dir=self.edge_dir,
                                          fanouts=self.fanouts)
        data_loader = DataLoader(dataset=dataset, batch_size=self.config.per_gpu_pretrain_batch_size,
                                 shuffle=True, pin_memory=True, drop_last=True,
                                 collate_fn=NodeSubGraphPairDataset.collate_fn)
        return data_loader
