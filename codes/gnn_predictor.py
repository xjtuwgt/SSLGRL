from codes.gnn_encoder import GraphSimSiamEncoder
from torch import nn
import torch


class NodeClassificationModel(nn.Module):
    def __init__(self, graph_encoder: GraphSimSiamEncoder, encoder_dim: int, num_of_classes: int,
                 fix_encoder: bool = True):
        super(NodeClassificationModel, self).__init__()
        self.encoder = graph_encoder
        self.predictor = nn.Linear(in_features=encoder_dim, out_features=num_of_classes)
        self.fix_encoder = fix_encoder

    @staticmethod
    def get_representation(sub_model: GraphSimSiamEncoder, batch: dict, cls_or_anchor: str, fix_encoder: bool = False):
        if fix_encoder:
            with torch.no_grad():
                graph_emb = sub_model.encode(batch=batch, cls_or_anchor=cls_or_anchor)
        else:
            graph_emb = sub_model.encode(batch=batch, cls_or_anchor=cls_or_anchor)
        return graph_emb

    def forward(self, batch, cls_or_anchor):
        graph_embed = self.get_representation(sub_model=self.encoder, batch=batch,
                                              fix_encoder=self.fix_encoder, cls_or_anchor=cls_or_anchor)
        pred_scores = self.predictor(graph_embed)
        return pred_scores
