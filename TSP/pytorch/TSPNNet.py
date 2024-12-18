import torch
import torch.nn as nn
import torch.nn.functional as F

from TSP.TSPGame import TSPGame


###################################
# TSPNNet Class
###################################
class TSPNNet(nn.Module):
    def __init__(self, game: TSPGame, args):
        super(TSPNNet, self).__init__()
        self.args = args
        self.num_nodes = game.getNumberOfNodes()
        # Actions: choose next node from remaining unvisited.
        # max action size = num_nodes-1
        self.action_size = game.getActionSize()

        # Node feature size: x,y coords + position in partial tour
        self.node_feature_size = 3
        self.hidden_dim = 128

        self.gc1 = nn.Linear(self.node_feature_size, self.hidden_dim)
        self.gc2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        self.fc_pi = nn.Linear(128, self.action_size)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, node_features, adjacency_matrix):
        x = self.graph_conv(node_features, adjacency_matrix, self.gc1)
        x = F.relu(x)
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        x = self.graph_conv(x, adjacency_matrix, self.gc2)
        x = F.relu(x)
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        x = x.mean(dim=1)  # mean pooling

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        pi = self.fc_pi(x)
        pi = F.softmax(pi, dim=1)

        v = self.fc_v(x)
        v = torch.tanh(v)

        return pi, v

    def graph_conv(self, node_features, adjacency_matrix, linear_layer):
        degrees = adjacency_matrix.sum(dim=-1, keepdim=True) + 1e-6
        norm_adj = adjacency_matrix / degrees
        agg_features = torch.bmm(norm_adj, node_features)
        out = linear_layer(agg_features)
        return out
