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
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout

        # Input layer: embed 4D coordinates into an embedding space
        self.input_layer = nn.Linear(4, self.embedding_dim)

        # Encoder: LSTM with configurable layers and dropout (if more than one layer)
        self.encoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
        )

        # Pointer: compute attention scores for each node from the encoder outputs
        self.pointer = nn.Linear(self.hidden_dim, 1)

        # Value head: a small MLP for state value estimation
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, node_features, adjacency):
        # node_features shape: (batch, num_nodes, 4)
        x = self.input_layer(node_features)  # (batch, num_nodes, embedding_dim)
        encoder_outputs, _ = self.encoder(x)  # (batch, num_nodes, hidden_dim)

        # Policy: for each node, get pointer logits
        policy_logits = self.pointer(encoder_outputs).squeeze(-1)  # (batch, num_nodes)

        # Value: average hidden states then pass through value head
        value = self.value_head(encoder_outputs.mean(dim=1)).squeeze(-1)  # (batch,)

        return policy_logits, value
