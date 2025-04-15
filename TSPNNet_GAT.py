import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from TSPGame import TSPGame  # Assuming TSPGame might be needed for dimensions

# Re-use the GraphAttentionLayer from TSPNNet_GraphPointer
# (Or copy it here if you prefer standalone files)
from TSPNNet_GraphPointer import GraphAttentionLayer


# Encoder using GAT layers
class GATEncoder(nn.Module):
    """Encoder using multiple GAT layers"""

    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.attentions = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # Optional LayerNorm

        current_dim = embedding_dim
        for i in range(num_layers):
            # Ensure embedding_dim is divisible by num_heads
            assert (
                embedding_dim % num_heads == 0
            ), "embedding_dim must be divisible by num_heads"
            head_dim = embedding_dim // num_heads

            # Create heads for the layer
            layer_attentions = [
                GraphAttentionLayer(
                    current_dim, head_dim, dropout=dropout, alpha=alpha, concat=True
                )
                for _ in range(num_heads)
            ]
            self.attentions.append(nn.ModuleList(layer_attentions))
            # LayerNorm after attention head concatenation
            self.layer_norms.append(nn.LayerNorm(embedding_dim))
            # Output dim is embedding_dim after concat
            current_dim = embedding_dim

        self.elu = nn.ELU()
        self.dropout = dropout

    def forward(self, node_features, adj=None):
        x = self.embedding(node_features)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Initial dropout

        for i, layer in enumerate(self.attentions):
            # Store residual for potential skip connection (optional)
            residual = x
            # Concatenate heads
            x_heads = torch.cat([att(x, adj) for att in layer], dim=-1)
            # Apply dropout after head concatenation
            x_heads = F.dropout(x_heads, p=self.dropout, training=self.training)

            # Add residual (optional, adjust projection if needed)
            # if x.shape == x_heads.shape: # Simple check
            #    x = residual + x_heads
            # else: # Need projection if dimensions changed
            x = x_heads  # Without skip connection for simplicity here

            # Apply LayerNorm and Activation
            x = self.layer_norms[i](x)
            x = self.elu(x)

        return x  # Final node embeddings [Batch, N, embedding_dim]


# Main GAT Network Class
class TSPNNet_GAT(nn.Module):
    def __init__(self, game: TSPGame, args):
        super(TSPNNet_GAT, self).__init__()
        self.args = args
        self.board_size = game.getNumberOfNodes()
        self.action_size = game.getActionSize()
        self.input_dim = 4  # x, y, tour_pos, visited (adjust if needed)

        # Components
        self.encoder = GATEncoder(
            input_dim=self.input_dim,
            embedding_dim=args.embedding_dim,
            num_heads=args.heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=0.2,  # LeakyReLU alpha for GAT
        )

        # Global pooling layer
        self.global_pool_type = args.get("global_pool", "mean")

        # Policy Head MLP (takes pooled graph embedding)
        policy_mlp_layers = []
        current_dim = args.embedding_dim  # Input is pooled embedding
        for _ in range(
            max(1, args.policy_layers) - 1
        ):  # Ensure at least one hidden layer if policy_layers > 1
            policy_mlp_layers.extend(
                [
                    nn.Linear(current_dim, args.policy_dim),
                    nn.ReLU(),  # Or get_activation() from args
                    nn.Dropout(args.dropout),
                ]
            )
            current_dim = args.policy_dim
        policy_mlp_layers.append(
            nn.Linear(current_dim, self.action_size)
        )  # Final output layer
        self.policy_head = nn.Sequential(*policy_mlp_layers)

        # Value Head MLP (takes pooled graph embedding)
        value_mlp_layers = []
        current_dim = args.embedding_dim  # Input is pooled embedding
        for _ in range(
            max(1, args.value_layers) - 1
        ):  # Ensure at least one hidden layer if value_layers > 1
            value_mlp_layers.extend(
                [
                    nn.Linear(current_dim, args.value_dim),
                    nn.ReLU(),  # Or get_activation() from args
                    nn.Dropout(args.dropout),
                ]
            )
            current_dim = args.value_dim
        value_mlp_layers.append(nn.Linear(current_dim, 1))  # Final output layer
        self.value_head = nn.Sequential(*value_mlp_layers)

    def forward(self, node_features, adj_matrix):
        # node_features: [Batch, N, input_dim]
        # adj_matrix: [Batch, N, N] (passed to encoder)

        # 1. Encode node features using GAT
        node_embeddings = self.encoder(
            node_features, adj_matrix
        )  # [Batch, N, embedding_dim]

        # 2. Global Pooling
        if self.global_pool_type == "mean":
            graph_embedding = node_embeddings.mean(dim=1)  # [Batch, embedding_dim]
        elif self.global_pool_type == "sum":
            graph_embedding = node_embeddings.sum(dim=1)
        elif self.global_pool_type == "max":
            graph_embedding = torch.max(node_embeddings, dim=1)[0]
        else:  # Default to mean
            graph_embedding = node_embeddings.mean(dim=1)

        # 3. Policy Prediction
        policy_logits = self.policy_head(graph_embedding)  # [Batch, action_size]

        # 4. Value Prediction
        value = self.value_head(graph_embedding)  # [Batch, 1]

        return policy_logits, value.squeeze(-1)  # Return logits and scalar value

    # --- Add parameter grouping methods for NNetWrapper ---
    def policy_parameters(self):
        """Returns an iterator over the policy head parameters."""
        return self.policy_head.parameters()

    def value_parameters(self):
        """Returns an iterator over the value head parameters."""
        return self.value_head.parameters()

    def shared_parameters(self):
        """Returns an iterator over parameters NOT in policy or value heads (i.e., the encoder)."""
        policy_ids = {id(p) for p in self.policy_parameters()}
        value_ids = {id(p) for p in self.value_parameters()}
        for p in self.parameters():
            if id(p) not in policy_ids and id(p) not in value_ids:
                yield p

    # -----------------------------------------------------
