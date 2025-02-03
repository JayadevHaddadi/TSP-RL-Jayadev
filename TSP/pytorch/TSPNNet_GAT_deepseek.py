import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.3):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.out_features = out_features
        
        # Linear transformation for node features
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        
        # Attention mechanism parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_features))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        B, N, _ = x.shape  # Batch size, Number of nodes, Features
        
        # Project features into multi-head space
        x_proj = self.linear(x).view(B, N, self.heads, self.out_features)
        x_proj = x_proj.permute(0, 2, 1, 3)  # [B, heads, N, out_features]
        
        # Calculate attention scores
        # Broadcast to all node pairs
        src_scores = (x_proj * self.att_src).sum(dim=-1, keepdim=True)  # [B, heads, N, 1]
        dst_scores = (x_proj * self.att_dst).sum(dim=-1, keepdim=True)  # [B, heads, N, 1]
        
        # Expand to create pairwise scores
        scores = src_scores + dst_scores.permute(0, 1, 3, 2)  # [B, heads, N, N]
        scores = self.leakyrelu(scores)
        
        # Apply adjacency mask
        adj = adj.unsqueeze(1)  # [B, 1, N, N]
        scores = scores.masked_fill(adj == 0, float('-inf'))
        
        # Compute attention weights
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Apply attention to node features
        out = torch.matmul(attn, x_proj)  # [B, heads, N, out_features]
        
        # Combine heads and return
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, N, heads, out_features]
        return out.view(B, N, self.heads * self.out_features)

class TSPNNet(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.args = args
        self.node_feature_size = 4  # Now expecting 4 features (added unvisited status)
        self.hidden_dim = 128
        self.heads = 4

        # GAT Layers
        self.gat1 = GATLayer(self.node_feature_size, self.hidden_dim//self.heads, self.heads)
        self.gat2 = GATLayer(self.hidden_dim, self.hidden_dim//self.heads, self.heads)
        
        # Policy/Value Heads
        self.fc1 = nn.Linear(self.hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_pi = nn.Linear(128, game.getActionSize())
        self.fc_v = nn.Linear(128, 1)

    def forward(self, node_features, adjacency):
        # Graph Attention
        x = F.elu(self.gat1(node_features, adjacency))
        x = F.dropout(x, self.args.dropout, training=self.training)
        x = F.elu(self.gat2(x, adjacency))
        x = F.dropout(x, self.args.dropout, training=self.training)
        
        # Readout
        x = x.mean(dim=1)
        
        # FC Layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.args.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.args.dropout, training=self.training)
        
        pi = F.softmax(self.fc_pi(x), dim=1)
        v = self.fc_v(x)
        return pi, v