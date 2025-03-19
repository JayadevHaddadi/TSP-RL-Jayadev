import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, game, args):
        super(TransformerModel, self).__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim  # Used as dim for feedforward
        self.num_layers = args.num_layers
        self.heads = args.heads
        self.dropout = args.dropout
        self.policy_layers = args.policy_layers
        self.policy_dim = args.policy_dim
        self.value_layers = args.value_layers
        self.value_dim = args.value_dim

        # Input layer: embed 2D coordinates into an embedding space
        self.input_layer = nn.Linear(4, self.embedding_dim)

        # Transformer encoder layer and encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.heads,
            dropout=self.dropout,
            dim_feedforward=self.hidden_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Policy head: build an MLP with policy_layers and policy_dim
        policy_modules = []
        input_size = self.embedding_dim
        for i in range(self.policy_layers - 1):
            policy_modules.append(nn.Linear(input_size, self.policy_dim))
            policy_modules.append(nn.ReLU())
            input_size = self.policy_dim
        policy_modules.append(nn.Linear(input_size, 1))
        self.policy_head = nn.Sequential(*policy_modules)

        # Value head: build an MLP with value_layers and value_dim
        value_modules = []
        input_size = self.embedding_dim
        for i in range(self.value_layers - 1):
            value_modules.append(nn.Linear(input_size, self.value_dim))
            value_modules.append(nn.ReLU())
            input_size = self.value_dim
        value_modules.append(nn.Linear(input_size, 1))
        value_modules.append(nn.Tanh())
        self.value_head = nn.Sequential(*value_modules)

    def forward(self, node_features, adjacency):
        # node_features shape: (batch, num_nodes, 2)
        x = self.input_layer(node_features)  # (batch, num_nodes, embedding_dim)

        # Rearrangement for transformer encoder: (num_nodes, batch, embedding_dim)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # back to (batch, num_nodes, embedding_dim)

        # Policy head: compute per-node scores
        policy_logits = self.policy_head(x).squeeze(-1)  # (batch, num_nodes)

        # Value head: aggregate node embeddings and compute state value
        value = self.value_head(x.mean(dim=1)).squeeze(-1)  # (batch,)

        return policy_logits, value
