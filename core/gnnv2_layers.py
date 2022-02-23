import torch
import torch.nn as nn
from dgl.nn.pytorch.utils import Identity
import dgl.function as fn
from torch import Tensor
from dgl.nn.functional import edge_softmax
from torch.nn import LayerNorm as layer_norm
from dgl.base import DGLError
from core.layers import PositionwiseFeedForward, small_init_gain


class RGDTLayer(nn.Module):
    def __init__(self,
                 in_ent_feats: int,
                 in_rel_feats: int,
                 out_ent_feats: int,
                 num_heads: int,
                 hop_num: int,
                 alpha: float = 0.15,
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 edge_drop: float = 0.1,
                 negative_slope: float = 0.2,
                 attn_degree_norm=True,
                 residual=True,
                 activation=None,
                 diff_head_tail=False,
                 ppr_diff=True):
        super(RGDTLayer, self).__init__()
        self._in_head_ent_feats, self._in_tail_ent_feats = in_ent_feats, in_ent_feats
        self._out_ent_feats = out_ent_feats
        self._in_rel_feats = in_rel_feats
        self._num_heads = num_heads
        self._hop_num = hop_num
        self._alpha = alpha
        self._attn_degree_norm = attn_degree_norm

        assert self._out_ent_feats % self._num_heads == 0
        self._head_dim = self._out_ent_feats // self._num_heads
        self.diff_head_tail = diff_head_tail

        if diff_head_tail:  # make different map
            self._ent_fc_head = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
            self._ent_fc_tail = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
        else:
            self._ent_fc = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)

        self._rel_fc = nn.Linear(self._in_rel_feats, self._num_heads * self._head_dim, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop

        self.attn = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)
        self.attn_activation = nn.PReLU(init=negative_slope)  # for attention computation

        if residual:
            if in_ent_feats != out_ent_feats:
                self.res_fc_ent = nn.Linear(in_ent_feats, self._num_heads * self._head_dim, bias=False)
            else:
                self.res_fc_ent = Identity()
        else:
            self.register_buffer('res_fc_ent', None)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_layer_norm = layer_norm(in_ent_feats)
        self.ff_layer_norm = layer_norm(self._num_heads * self._head_dim)
        self.feed_forward_layer = PositionwiseFeedForward(model_dim=self._num_heads * self._head_dim,
                                                          d_hidden=4 * self._num_heads * self._head_dim)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.reset_parameters()
        self.activation = activation
        self.ppr_diff = ppr_diff

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        # gain = nn.init.calculate_gain('relu')
        gain = small_init_gain(d_in=self._in_head_ent_feats, d_out=self._in_head_ent_feats)
        if self.diff_head_tail:
            nn.init.xavier_normal_(self._ent_fc_head.weight, gain=gain)
            nn.init.xavier_normal_(self._ent_fc_tail.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self._ent_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        nn.init.xavier_normal_(self._rel_fc.weight, gain=gain)
        if isinstance(self.res_fc_ent, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_ent.weight, gain=gain)

    def forward(self, graph, ent_feat: Tensor, rel_feat: Tensor, get_attention=False):
        ent_head = ent_tail = self.graph_layer_norm(ent_feat)
        ent_head = self.feat_drop(ent_head)
        ent_tail = self.feat_drop(ent_tail)
        rel_emb = self.feat_drop(rel_feat)
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               ' Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')
            if self.diff_head_tail:
                feat_head = self._ent_fc_head(ent_head).view(-1, self._num_heads, self._head_dim)
                feat_tail = self._ent_fc_tail(ent_tail).view(-1, self._num_heads, self._head_dim)
            else:
                feat_head = feat_tail = self._ent_fc(ent_head).view(-1, self._num_heads, self._head_dim)

            graph.srcdata.update({'ft': feat_head})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'et': feat_tail})
            graph.apply_edges(fn.u_add_v('ft', 'et', 'e'))
            feat_rel = self._rel_fc(rel_emb).view(-1, self._num_heads, self._head_dim)
            feat_rel = feat_rel[graph.edata['rid']]
            e = self.attn_activation(graph.edata.pop('e') + feat_rel)  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # if self._attn_degree_norm:
            #     graph.edata.update({'e': e})
            #     graph.apply_edges(fn.e_mul_v('e', 'log_in', 'e'))
            #     e = graph.edata.pop('e')
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                a_value = torch.zeros_like(e)
                a_value[eids] = edge_softmax(graph, e[eids], eids=eids)
            else:
                a_value = edge_softmax(graph, e)

            if self.ppr_diff:
                # graph.edata['a'] = edge_softmax(graph, e)
                graph.edata['a'] = a_value
                rst = self.ppr_estimation(graph=graph)
            else:
                # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                graph.edata['a'] = self.attn_drop(a_value)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                 fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
            # residual
            if self.res_fc_ent is not None:
                resval = self.res_fc_ent(ent_feat).view(ent_feat.shape[0], -1, self._head_dim)
                rst = self.feat_drop(rst) + resval

            rst = rst.flatten(1)
            # +++++++++++++++++++++++++++++++++++++++
            ff_rst = self.feed_forward_layer(self.feat_drop(self.ff_layer_norm(rst)))
            rst = self.feat_drop(ff_rst) + rst  # residual
            # +++++++++++++++++++++++++++++++++++++++
            # activation
            if self.activation:
                rst = self.activation(rst)
            # +++++++++++++++++++++++++++++++++++++++
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

    def ppr_estimation(self, graph):
        with graph.local_scope():
            graph = graph.local_var()
            feat_0 = graph.srcdata.pop('ft')
            feat = feat_0.clone()
            attentions = graph.edata.pop('a')
            for _ in range(self._hop_num):
                graph.srcdata['h'] = self.feat_drop(feat)
                graph.edata['a_temp'] = self.attn_drop(attentions)
                graph.update_all(fn.u_mul_e('h', 'a_temp', 'm'), fn.sum('m', 'h'))
                feat = graph.dstdata.pop('h')
                feat = (1.0 - self._alpha) * self.feat_drop(feat) + self._alpha * feat_0
            return feat


class GDTLayer(nn.Module):
    def __init__(self,
                 in_ent_feats: int,
                 out_ent_feats: int,
                 num_heads: int,
                 hop_num: int,
                 alpha: float = 0.15,
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 edge_drop: float = 0.1,
                 negative_slope: float = 0.2,
                 attn_degree_norm=True,
                 residual=True,
                 activation=None,
                 diff_head_tail=False,
                 ppr_diff=True):
        super(GDTLayer, self).__init__()
        self._in_head_ent_feats, self._in_tail_ent_feats = in_ent_feats, in_ent_feats
        self._out_ent_feats = out_ent_feats
        self._num_heads = num_heads
        self._hop_num = hop_num
        self._alpha = alpha
        self._attn_degree_norm = attn_degree_norm

        assert self._out_ent_feats % self._num_heads == 0
        self._head_dim = self._out_ent_feats // self._num_heads
        self.diff_head_tail = diff_head_tail

        if diff_head_tail:  # make different map
            self._ent_fc_head = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
            self._ent_fc_tail = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)
        else:
            self._ent_fc = nn.Linear(self._in_head_ent_feats, self._num_heads * self._head_dim, bias=False)

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop

        self.attn = nn.Parameter(torch.FloatTensor(1, self._num_heads, self._head_dim), requires_grad=True)
        self.attn_activation = nn.PReLU(init=negative_slope)  # for attention computation

        if residual:
            if in_ent_feats != out_ent_feats:
                self.res_fc_ent = nn.Linear(in_ent_feats, self._num_heads * self._head_dim, bias=False)
            else:
                self.res_fc_ent = Identity()
        else:
            self.register_buffer('res_fc_ent', None)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_layer_norm = layer_norm(in_ent_feats)
        self.ff_layer_norm = layer_norm(self._num_heads * self._head_dim)
        self.feed_forward_layer = PositionwiseFeedForward(model_dim=self._num_heads * self._head_dim,
                                                          d_hidden=4 * self._num_heads * self._head_dim)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.reset_parameters()
        self.activation = activation
        self.ppr_diff = ppr_diff

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        # gain = nn.init.calculate_gain('relu')
        gain = small_init_gain(d_in=self._in_head_ent_feats, d_out=self._in_head_ent_feats)
        if self.diff_head_tail:
            nn.init.xavier_normal_(self._ent_fc_head.weight, gain=gain)
            nn.init.xavier_normal_(self._ent_fc_tail.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self._ent_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc_ent, nn.Linear):
            nn.init.xavier_normal_(self.res_fc_ent.weight, gain=gain)

    def forward(self, graph, ent_feat: Tensor, get_attention=False):
        ent_head = ent_tail = self.graph_layer_norm(ent_feat)
        ent_head = self.feat_drop(ent_head)
        ent_tail = self.feat_drop(ent_tail)

        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               ' Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')
            if self.diff_head_tail:
                feat_head = self._ent_fc_head(ent_head).view(-1, self._num_heads, self._head_dim)
                feat_tail = self._ent_fc_tail(ent_tail).view(-1, self._num_heads, self._head_dim)
            else:
                feat_head = feat_tail = self._ent_fc(ent_head).view(-1, self._num_heads, self._head_dim)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            graph.srcdata.update({'ft': feat_head})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'et': feat_tail})
            graph.apply_edges(fn.u_add_v('ft', 'et', 'e'))
            e = self.attn_activation(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # if self._attn_degree_norm:
            #     graph.edata.update({'e': e})
            #     graph.apply_edges(fn.e_mul_v('e', 'log_in', 'e'))
            #     e = graph.edata.pop('e')
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                a_value = torch.zeros_like(e)
                a_value[eids] = edge_softmax(graph, e[eids], eids=eids)
            else:
                a_value = edge_softmax(graph, e)

            if self.ppr_diff:
                # graph.edata['a'] = edge_softmax(graph, e)
                graph.edata['a'] = a_value
                rst = self.ppr_estimation(graph=graph)
            else:
                # graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
                graph.edata['a'] = self.attn_drop(a_value)
                graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                                 fn.sum('m', 'ft'))
                rst = graph.dstdata['ft']
            # residual
            if self.res_fc_ent is not None:
                resval = self.res_fc_ent(ent_feat).view(ent_feat.shape[0], -1, self._head_dim)
                rst = self.feat_drop(rst) + resval

            rst = rst.flatten(1)
            # +++++++++++++++++++++++++++++++++++++++
            ff_rst = self.feed_forward_layer(self.feat_drop(self.ff_layer_norm(rst)))
            rst = self.feat_drop(ff_rst) + rst  # residual
            # +++++++++++++++++++++++++++++++++++++++
            # activation
            if self.activation:
                rst = self.activation(rst)
            # +++++++++++++++++++++++++++++++++++++++
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

    def ppr_estimation(self, graph):
        with graph.local_scope():
            graph = graph.local_var()
            feat_0 = graph.srcdata.pop('ft')
            feat = feat_0.clone()
            attentions = graph.edata.pop('a')
            for _ in range(self._hop_num):
                graph.srcdata['h'] = self.feat_drop(feat)
                graph.edata['a_temp'] = self.attn_drop(attentions)
                graph.update_all(fn.u_mul_e('h', 'a_temp', 'm'), fn.sum('m', 'h'))
                feat = graph.dstdata.pop('h')
                feat = (1.0 - self._alpha) * self.feat_drop(feat) + self._alpha * feat_0
            return feat
