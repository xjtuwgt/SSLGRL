import torch, math
from torch import Tensor, LongTensor
from torch import nn
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, num: int, dim: int, project_dim: int = None):
        super(EmbeddingLayer, self).__init__()
        self.num = num
        self.dim = dim
        self.proj_dim = project_dim
        self.embedding = nn.Embedding(num_embeddings=num, embedding_dim=dim)
        if self.proj_dim is not None and self.proj_dim > 0:
            self.projection = torch.nn.Linear(self.dim, self.proj_dim, bias=False)
        else:
            self.projection = Identity()

    def init_with_tensor(self, data: Tensor, freeze=False):
        self.embedding = nn.Embedding.from_pretrained(embeddings=data, freeze=freeze)

    def init(self, emb_init=0.1):
        """Initializing the embeddings.
        Parameters
        ----------
        emb_init : float
            The initial embedding range should be [-emb_init, emb_init].
        """
        nn.init.xavier_normal_(self.embedding.weight, emb_init)
        gain = nn.init.calculate_gain('relu')
        if isinstance(self.projection, nn.Linear):
            nn.init.xavier_normal_(self.projection.weight, gain=gain)

    def _embed(self, embeddings):
        embeddings = self.projection(embeddings)
        return embeddings

    def forward(self, indexes: LongTensor):
        embed_data = self._embed(self.embedding(indexes))
        return embed_data


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, model_dim, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = d_hidden
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.w_2 = nn.Linear(d_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

    def init(self):
        # gain = nn.init.calculate_gain('relu')
        gain = small_init_gain(d_in=self.model_dim, d_out=self.hidden_dim)
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
        gain = small_init_gain(d_in=self.hidden_dim, d_out=self.model_dim)
        nn.init.xavier_normal_(self.w_2.weight, gain=gain)


def small_init_gain(d_in, d_out):
    return 2.0 / math.sqrt(d_in + 4.0 * d_out)