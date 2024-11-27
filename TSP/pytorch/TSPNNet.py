import torch
import torch.nn as nn
import torch.nn.functional as F

class TSPNNet(nn.Module):
    def __init__(self, game, args):
        super(TSPNNet, self).__init__()
        self.args = args

        # Game parameters
        self.num_nodes = game.getBoardSize()[0]
        self.action_size = game.getActionSize()

        # Node feature size (e.g., x, y, tour_position)
        self.node_feature_size = 3

        # Hidden dimensions
        self.hidden_dim = 128

        # Graph convolutional layers
        self.gc1 = nn.Linear(self.node_feature_size, self.hidden_dim)
        self.gc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Fully connected layers after graph convolutions
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        # Policy head
        self.fc_pi = nn.Linear(128, self.action_size)

        # Value head
        self.fc_v = nn.Linear(128, 1)

    def forward(self, node_features, adjacency_matrix):
        """
        node_features: tensor of shape [batch_size, num_nodes, node_feature_size]
        adjacency_matrix: tensor of shape [batch_size, num_nodes, num_nodes]
        """
        batch_size = node_features.size(0)

        # First graph convolutional layer
        x = self.graph_conv(node_features, adjacency_matrix, self.gc1)
        x = F.relu(x)
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        # Second graph convolutional layer
        x = self.graph_conv(x, adjacency_matrix, self.gc2)
        x = F.relu(x)
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        # Aggregate node features by mean pooling
        x = x.mean(dim=1)  # Shape: [batch_size, hidden_dim]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        # Policy head
        pi = self.fc_pi(x)  # Shape: [batch_size, action_size]
        pi = F.softmax(pi, dim=1)

        # Value head
        v = self.fc_v(x)  # Shape: [batch_size, 1]
        v = torch.tanh(v)

        return pi, v

    def graph_conv(self, node_features, adjacency_matrix, linear_layer):
        """
        Simplified graph convolution operation.
        """
        degrees = adjacency_matrix.sum(dim=-1, keepdim=True) + 1e-6
        norm_adj = adjacency_matrix / degrees

        # Aggregate neighboring node features
        agg_features = torch.bmm(norm_adj, node_features)

        # Apply linear transformation
        out = linear_layer(agg_features)

        return out
