import torch
import torch.nn as nn
import torch.nn.functional as F

class TSPNNet_Pointer(nn.Module):
    """
    Minimal pointer-like net: 
    - We'll embed each node (with x,y plus partial-tour position).
    - Then we'll have a "decoder" query embedding to get a distribution over unvisited nodes.
    """

    def __init__(self, game, args):
        super(TSPNNet_Pointer, self).__init__()
        self.args = args
        self.num_nodes = game.getNumberOfNodes()
        self.action_size = game.getActionSize()

        self.node_feature_size = 4
        hidden_dim = getattr(args, "num_channels", 128)

        # 1) Node embedding
        self.embedding = nn.Linear(self.node_feature_size, hidden_dim)

        # 2) "Decoder LSTM" or single query vector
        self.decoder_rnn = nn.LSTMCell(hidden_dim, hidden_dim)

        # Optionally, have a single "query" parameter
        self.query = nn.Parameter(torch.randn(1, hidden_dim))

        # 3) Attn projection
        self.attn_W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.randn(hidden_dim))  # for dot with tanh

        # 4) Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, node_features, adjacency_matrix):
        """
        node_features: [B, N, 3]
        adjacency_matrix: [B, N, N] (not used strongly here).
        We'll produce:
          pi: [B, N] distribution
          v:  [B, 1] leftover distance
        """
        B, N, _ = node_features.size()

        # 1) embed nodes
        node_emb = self.embedding(node_features)  # [B, N, hidden_dim]
        node_emb = F.relu(node_emb)

        # 2) decode step
        # We'll treat self.query as the decoder's hidden/cell init
        query = self.query.expand(B, -1)  # [B, hidden_dim]
        hx, cx = self.decoder_rnn(query)  # you might do hx,cx = self.decoder_rnn(query,h0)
        # but for simplicity let's do no initial hidden => 0

        # 3) attention 
        #  atn_i = v^T tanh( W * node_emb + hx )
        #  => shape [B,N]
        Wemb = self.attn_W(node_emb)  # [B, N, hidden_dim]
        hx_reshaped = hx.unsqueeze(1).expand(B, N, hx.size(-1))  # [B,N,hidden_dim]

        sum_emb = torch.tanh(Wemb + hx_reshaped)  # [B,N,hidden_dim]
        # dot with v
        attn_logits = torch.einsum("bnh,h->bn", sum_emb, self.v)  # => [B,N]
        # mask unvisited? We'll rely on MCTS for that, or we can softmax then multiply by valid moves

        pi = F.softmax(attn_logits, dim=-1)  # [B,N]

        # 4) leftover distance as value => use hx
        v = self.value_head(hx)  # [B,1]

        return pi, v
