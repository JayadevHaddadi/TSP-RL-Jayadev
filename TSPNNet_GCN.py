import torch
import torch.nn as nn
import torch.nn.functional as F

from TSPGame import TSPGame


class TSPNNet(nn.Module):
    def __init__(self, game: TSPGame, args):
        super(TSPNNet, self).__init__()
        self.args = args
        self.board_size = game.getNumberOfNodes()
        self.action_size = game.getActionSize()

        # Input dimensions
        self.input_dim = 4  # x, y, tour_position, visited
        self.embedding_dim = args.embedding_dim

        # Initial embedding layer
        self.embedding = nn.Linear(self.input_dim, self.embedding_dim)

        # Edge features if enabled
        self.use_edge_features = args.use_edge_features
        if self.use_edge_features:
            self.edge_embedding = nn.Linear(1, args.edge_dim)
            self.edge_proj = nn.Linear(args.edge_dim, self.embedding_dim)

        # Graph Convolution Layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        current_dim = self.embedding_dim
        for _ in range(args.num_layers):
            # GCN layer
            layer = nn.Linear(current_dim, args.num_channels)
            self.conv_layers.append(layer)

            # Layer normalization if enabled
            if args.layer_norm:
                self.layer_norms.append(nn.LayerNorm(args.num_channels))

            current_dim = args.num_channels

        # Batch normalization if enabled
        self.use_batch_norm = args.batch_norm
        if self.use_batch_norm:
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(args.num_channels) for _ in range(args.num_layers)]
            )

        # Skip connections if enabled
        self.use_skip = args.skip_connections
        if self.use_skip and self.embedding_dim != args.num_channels:
            self.skip_proj = nn.Linear(self.embedding_dim, args.num_channels)

        # Global pooling
        self.global_pool = args.global_pool

        # Readout MLP
        readout_layers = []
        current_dim = args.num_channels

        # Handle the case where readout_layers is 1
        if args.readout_layers <= 1:
            self.readout = nn.Identity()
        else:
            for _ in range(args.readout_layers - 1):
                readout_layers.extend(
                    [
                        nn.Linear(current_dim, args.readout_dim),
                        self.get_activation(),
                        nn.Dropout(args.dropout),
                    ]
                )
                current_dim = args.readout_dim
            self.readout = nn.Sequential(*readout_layers)

        # Policy head (ensure at least one layer)
        policy_layers = []
        current_dim = args.readout_dim if args.readout_layers > 1 else args.num_channels
        if args.policy_layers <= 1:
            policy_layers.append(nn.Linear(current_dim, self.action_size))
        else:
            for _ in range(args.policy_layers - 1):
                policy_layers.extend(
                    [
                        nn.Linear(current_dim, args.policy_dim),
                        self.get_activation(),
                        nn.Dropout(args.dropout),
                    ]
                )
                current_dim = args.policy_dim
            policy_layers.append(nn.Linear(current_dim, self.action_size))
        self.policy_head = nn.Sequential(*policy_layers)

        # Value head (ensure at least one layer)
        value_layers = []
        current_dim = args.readout_dim if args.readout_layers > 1 else args.num_channels
        if args.value_layers <= 1:
            value_layers.append(nn.Linear(current_dim, 1))
        else:
            for _ in range(args.value_layers - 1):
                value_layers.extend(
                    [
                        nn.Linear(current_dim, args.value_dim),
                        self.get_activation(),
                        nn.Dropout(args.dropout),
                    ]
                )
                current_dim = args.value_dim
            value_layers.append(nn.Linear(current_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.args.init_type == "kaiming":
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity=self.args.activation
                )
            elif self.args.init_type == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif self.args.init_type == "orthogonal":
                nn.init.orthogonal_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

            # Apply initialization scale
            module.weight.data *= self.args.init_scale

    def get_activation(self):
        if self.args.activation == "relu":
            return nn.ReLU()
        elif self.args.activation == "gelu":
            return nn.GELU()
        elif self.args.activation == "elu":
            return nn.ELU()
        else:
            return nn.ReLU()  # default

    def graph_conv(self, node_features, adjacency_matrix, linear_layer, layer_idx):
        """Enhanced graph convolution with edge features and normalization"""
        # Degree-based normalization
        degrees = adjacency_matrix.sum(dim=-1, keepdim=True) + 1e-6
        norm_adj = adjacency_matrix / degrees

        # Edge features if enabled
        if self.use_edge_features:
            # Create edge features (could be distances or other metrics)
            edge_features = self.edge_embedding(adjacency_matrix.unsqueeze(-1).float())
            edge_weights = self.edge_proj(edge_features)
            # Combine with adjacency (handle broadcasting properly)
            norm_adj = norm_adj.unsqueeze(-1) * edge_weights
            # Sum over edge feature dimension
            norm_adj = norm_adj.sum(dim=-1)

        # Message passing
        agg_features = torch.bmm(norm_adj, node_features)
        out = linear_layer(agg_features)

        # Apply normalizations (handle shapes carefully)
        if self.use_batch_norm:
            shape = out.shape
            out = out.view(-1, shape[-1])
            out = self.batch_norms[layer_idx](out)
            out = out.view(*shape)

        if self.args.layer_norm:
            out = self.layer_norms[layer_idx](out)

        return out

    def forward(self, node_features, adjacency_matrix):
        # Initial embedding
        x = self.embedding(node_features)

        # Store initial features for skip connection
        initial_features = x
        if self.use_skip and hasattr(self, "skip_proj"):
            initial_features = self.skip_proj(initial_features)

        # Graph convolution layers
        for i, conv_layer in enumerate(self.conv_layers):
            # Apply graph convolution
            out = self.graph_conv(x, adjacency_matrix, conv_layer, i)

            # Skip connection if enabled
            if self.use_skip:
                out = out + initial_features

            # Activation
            out = self.get_activation()(out)

            # Dropout
            out = F.dropout(out, p=self.args.dropout, training=self.training)

            x = out

        # Global pooling
        if self.global_pool == "mean":
            pooled = torch.mean(x, dim=1)
        elif self.global_pool == "sum":
            pooled = torch.sum(x, dim=1)
        else:  # max
            pooled = torch.max(x, dim=1)[0]

        # Readout MLP
        features = self.readout(pooled)

        # Policy and value heads
        pi = self.policy_head(features)
        v = self.value_head(features)

        return pi, v.squeeze(-1)
