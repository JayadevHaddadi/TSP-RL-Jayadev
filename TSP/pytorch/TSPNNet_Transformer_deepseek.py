import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        self.args = args
        self.node_feature_size = 3
        self.hidden_dim = 128
        self.nhead = 4
        self.num_layers = 3

        # Embedding
        self.embedding = nn.Linear(self.node_feature_size, self.hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=512,
            dropout=args.dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # Heads
        self.fc_pi = nn.Linear(self.hidden_dim, game.getActionSize())
        self.fc_v = nn.Linear(self.hidden_dim, 1)

    def forward(self, node_features, adjacency):
        # Embed and add positional encoding
        x = self.embedding(node_features)  # [B,N,feat]
        x = x.permute(1, 0, 2)  # [N,B,feat] for transformer
        
        # Transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [B,N,feat]
        
        # Pooling
        x = x.mean(dim=1)  # Global average pooling
        
        # Outputs
        pi = F.softmax(self.fc_pi(x), dim=1)
        v = self.fc_v(x)
        return pi, v