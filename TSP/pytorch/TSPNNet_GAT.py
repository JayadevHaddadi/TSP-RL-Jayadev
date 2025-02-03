import torch
import torch.nn as nn
import torch.nn.functional as F

class TSPNNet_GAT(nn.Module):
    """
    A simple GAT-based approach for partial TSP tours.
    - node_features: (num_nodes, 3) => x, y, tour_position
    - adjacency_matrix: edges between consecutive visited nodes

    We'll add a self-attention layer that can consider all pairs (or neighbors).
    """

    def __init__(self, game, args):
        super(TSPNNet_GAT, self).__init__()
        self.args = args
        self.num_nodes = game.getNumberOfNodes()
        self.action_size = game.getActionSize()

        # Node feature size = 3
        self.node_feature_size = 3
        hidden_dim = getattr(args, "num_channels", 128)

        # GAT parameters
        self.heads = 4  # number of attention heads
        self.hidden_dim = hidden_dim

        # 1) Input linear transform
        self.fc_input = nn.Linear(self.node_feature_size, hidden_dim, bias=False)

        # 2) Learnable vectors for attention: 
        #    "a" vectors in typical GAT approach for source & target
        self.att_src = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        self.att_dst = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(self.att_src.data)
        nn.init.xavier_uniform_(self.att_dst.data)

        # 3) Possibly do multi-head by repeating. 
        #    For brevity, let's do single-head or 2-head version

        # 4) Another GAT layer or a feed-forward
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.att_src2 = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        self.att_dst2 = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(self.att_src2.data)
        nn.init.xavier_uniform_(self.att_dst2.data)

        # After GAT layers, we pool node embeddings => feed to MLP
        self.mlp1 = nn.Linear(hidden_dim, 256)
        self.mlp2 = nn.Linear(256, 128)

        self.fc_pi = nn.Linear(128, self.action_size)
        self.fc_v = nn.Linear(128, 1)

    def forward(self, node_features, adjacency_matrix):
        """
        node_features: [batch_size, num_nodes, 3]
        adjacency_matrix: [batch_size, num_nodes, num_nodes], 
                          edges for partial tour or possibly we can ignore it 
                          since GAT doesn't always need adjacency. 
        """
        B, N, F_in = node_features.size()

        # 1) Linear transform node features
        x = self.fc_input(node_features)  # shape [B, N, hidden_dim]

        # 2) GAT attention #1
        x = self.gat_layer(x, adjacency_matrix,
                           self.att_src, self.att_dst,
                           activation=F.relu)

        # 3) GAT attention #2
        x = self.gat_layer(x, adjacency_matrix,
                           self.att_src2, self.att_dst2,
                           activation=F.relu)

        # 4) Mean pool over nodes
        x = x.mean(dim=1)  # [B, hidden_dim]

        # 5) MLP
        x = F.relu(self.mlp1(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.mlp2(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)

        pi = self.fc_pi(x)
        pi = F.softmax(pi, dim=1)

        v = self.fc_v(x)  # leftover distance

        return pi, v

    def gat_layer(self, x, adj, att_src, att_dst, activation=F.relu):
        """
        A single GAT-like layer:
        x: [B, N, hidden_dim]
        adj: [B, N, N] => we can use it to mask out attention on edges that don't exist
        att_src, att_dst: [hidden_dim, 1]
        """
        B, N, C = x.size()

        # 1) Expand for pairwise combination
        #    x_i => shape [B, N, 1, C]
        #    x_j => shape [B, 1, N, C]
        x_i = x.unsqueeze(2)  # [B, N, 1, C]
        x_j = x.unsqueeze(1)  # [B, 1, N, C]

        # 2) Compute e_ij = LeakyReLU( x_i * a_src + x_j * a_dst )
        #    shapes => x_i:[B,N,1,C], att_src:[C,1] => x_i * att_src => [B,N,1,1]
        alpha_i = torch.matmul(x_i, att_src)  # [B, N, 1, 1]
        alpha_j = torch.matmul(x_j, att_dst)  # [B, 1, N, 1]
        e = alpha_i + alpha_j  # [B, N, N, 1]
        e = F.leaky_relu(e, negative_slope=0.2).squeeze(-1)  # [B, N, N]

        # 3) Mask using adjacency => set -inf where adj=0
        #    or we can do adjacency>0 if partial
        if adj is not None:
            mask = (adj > 0).float()  # shape [B,N,N]
        else:
            mask = torch.ones_like(e)  # fully connected
        e = e.masked_fill(mask < 0.5, float('-inf'))

        # 4) Softmax
        alpha = F.softmax(e, dim=-1)  # [B, N, N]

        # 5) Weighted sum
        #   out_i = sum_j alpha_ij * x_j
        #   x_j: [B,1,N,C], alpha: [B,N,N] => we do batch matmul
        alpha = alpha.unsqueeze(-1)  # [B, N, N, 1]
        x_j = x_j  # [B, 1, N, C] => we want [B,N,N,C], so do x_j.expand?
        x_j = x_j.expand(B, N, N, C)
        out = alpha * x_j  # [B,N,N,C]
        out = out.sum(dim=2)  # [B,N,C]

        # 6) activation
        out = activation(out)
        # optional dropout:
        out = F.dropout(out, p=self.args.dropout, training=self.training)

        return out
