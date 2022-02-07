from core.gnn_layers import GDTLayer, RGDTLayer
from torch import Tensor
from core.siamese_network import SimSiam
from core.layers import EmbeddingLayer
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_cosine_with_hard_restarts_schedule_with_warmup
import logging


class GDTEncoder(nn.Module):
    def __init__(self, config):
        super(GDTEncoder, self).__init__()
        self.config = config
        self.node_embed_layer = EmbeddingLayer(num=self.config.node_number, dim=self.config.node_emb_dim)
        if self.config.relation_encoder:
            self.relation_embed_layer = EmbeddingLayer(num=self.config.relation_number,
                                                       dim=self.config.relation_emb_dim)
        else:
            self.relation_embed_layer = None
        if self.config.arw_position:
            arw_position_num = self.config.sub_graph_hop_num + 2
            self.arw_position_embed_layer = EmbeddingLayer(num=arw_position_num,
                                                           dim=self.config.node_emb_dim)
        self.graph_encoder = nn.ModuleList()
        if self.config.relation_encoder:
            self.graph_encoder.append(module=RGDTLayer(in_ent_feats=self.config.node_emb_dim,
                                                       in_rel_feats=self.config.relation_emb_dim,
                                                       out_ent_feats=self.config.hidden_dim,
                                                       num_heads=self.config.head_num,
                                                       hop_num=self.config.gnn_hop_num,
                                                       alpha=self.config.alpha,
                                                       feat_drop=self.config.feat_drop,
                                                       attn_drop=self.config.attn_drop,
                                                       residual=self.config.residual,
                                                       diff_head_tail=self.config.diff_head_tail,
                                                       ppr_diff=self.config.ppr_diff))
        else:
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.node_emb_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      residual=self.config.residual,
                                                      diff_head_tail=self.config.diff_head_tail,
                                                      ppr_diff=self.config.ppr_diff))
        for _ in range(1, self.config.layers):
            self.graph_encoder.append(module=GDTLayer(in_ent_feats=self.config.hidden_dim,
                                                      out_ent_feats=self.config.hidden_dim,
                                                      num_heads=self.config.head_num,
                                                      hop_num=self.config.gnn_hop_num,
                                                      alpha=self.config.alpha,
                                                      feat_drop=self.config.feat_drop,
                                                      attn_drop=self.config.attn_drop,
                                                      residual=self.config.residual,
                                                      diff_head_tail=self.config.diff_head_tail,
                                                      ppr_diff=self.config.ppr_diff))

    def init(self, graph_node_emb: Tensor = None, graph_rel_emb: Tensor = None, pos_emb: Tensor = None,
             node_freeze=False, rel_freeze=False, pos_freeze=False):
        if graph_node_emb is not None:
            self.node_embed_layer.init_with_tensor(data=graph_node_emb, freeze=node_freeze)
            logging.info('Initializing node features with pretrained embeddings')
        else:
            self.node_embed_layer.init()
        if self.config.relation_encoder:
            if graph_rel_emb is not None:
                self.relation_embed_layer.init_with_tensor(data=graph_rel_emb, freeze=rel_freeze)
                logging.info('Initializing relation embedding with pretrained embeddings')
            else:
                self.relation_embed_layer.init()
        if self.config.arw_position:
            if pos_emb is not None:
                self.arw_position_embed_layer.init_with_tensor(data=pos_emb, freeze=pos_freeze)
            else:
                self.arw_position_embed_layer.init()

    def forward(self, batch_g_pair, cls_or_anchor='cls'):
        if self.config.relation_encoder:
            return self.rel_forward(batch_g_pair=batch_g_pair, cls_or_anchor=cls_or_anchor)
        else:
            return self.no_rel_forward(batch_g_pair=batch_g_pair, cls_or_anchor=cls_or_anchor)

    def rel_forward(self, batch_g_pair, cls_or_anchor='cls'):
        batch_g = batch_g_pair[0]
        ent_ids = batch_g.ndata['nid']
        rel_ids = batch_g.edata['rid']
        ent_features = self.node_embed_layer(ent_ids)
        rel_features = self.relation_embed_layer(rel_ids)
        if self.config.arw_position:
            arw_positions = batch_g.ndata['n_rw_label']
            arw_pos_embed = self.arw_position_embed_layer(arw_positions)
            ent_features = ent_features + arw_pos_embed
        with batch_g.local_scope():
            h = ent_features
            for _ in range(self.config.layers):
                if _ == 0:
                    h = self.graph_encoder[_](batch_g, h, rel_features)
                else:
                    h = self.graph_encoder[_](batch_g, h)
            if cls_or_anchor == 'cls':
                batch_node_ids = batch_g_pair[1]
            elif cls_or_anchor == 'anchor':
                batch_node_ids = batch_g_pair[2]
            else:
                raise '{} is not supported'.format(cls_or_anchor)
            batch_graph_embed = h[batch_node_ids]
            return batch_graph_embed

    def no_rel_forward(self, batch_g_pair, cls_or_anchor='cls'):
        batch_g = batch_g_pair[0]
        ent_ids = batch_g.ndata['nid']
        ent_features = self.node_embed_layer(ent_ids)
        if self.config.arw_position:
            arw_positions = batch_g.ndata['n_rw_label']
            arw_pos_embed = self.arw_position_embed_layer(arw_positions)
            ent_features = ent_features + arw_pos_embed
        with batch_g.local_scope():
            h = ent_features
            for _ in range(self.config.layers):
                h = self.graph_encoder[_](batch_g, h)
            if cls_or_anchor == 'cls':
                batch_node_ids = batch_g_pair[1]
            elif cls_or_anchor == 'anchor':
                batch_node_ids = batch_g_pair[2]
            else:
                raise '{} is not supported'.format(cls_or_anchor)
            batch_graph_embed = h[batch_node_ids]
            return batch_graph_embed


class GraphSimSiamEncoder(nn.Module):
    def __init__(self, config):
        super(GraphSimSiamEncoder, self).__init__()
        self.config = config
        graph_encoder = GDTEncoder(config=config)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_siam_encoder = SimSiam(base_encoder=graph_encoder,
                                          base_encoder_out_dim=self.config.hidden_dim,
                                          dim=self.config.siam_dim)

    def init(self, graph_node_emb: Tensor = None, graph_rel_emb: Tensor = None, pos_emb: Tensor = None,
             node_freeze=False, rel_freeze=False, pos_freeze=False):
        self.graph_siam_encoder.graph_encoder.init(graph_node_emb=graph_node_emb,
                                                   graph_rel_emb=graph_rel_emb,
                                                   pos_emb=pos_emb,
                                                   node_freeze=node_freeze,
                                                   rel_freeze=rel_freeze,
                                                   pos_freeze=pos_freeze)

    def forward(self, batch, cls_or_anchor='cls'):
        p1, p2, z1, z2 = self.graph_siam_encoder(batch['batch_graph_1'], batch['batch_graph_2'], cls_or_anchor)
        return p1, p2, z1, z2

    def encode(self, batch, cls_or_anchor='cls'):
        embed = self.graph_siam_encoder.encode(x=batch['batch_graph'], cls_or_anchor=cls_or_anchor)
        return embed

    def prepare_optimizer_scheduler(self, total_steps):
        "Prepare optimizer and schedule (linear warmup and decay)"
        optimization_params = self.parameters()
        optimizer = AdamW(optimization_params, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        if self.config.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.config.warmup_steps,
                                                                           num_training_steps=total_steps)
        else:
            raise '{} is not supported'.format(self.config.lr_scheduler)
        return optimizer, scheduler
