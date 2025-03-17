import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    def __init__(self, d_model, args):
        super(ConformerBlock, self).__init__()
        # First feed forward module
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=args.heads, dropout=args.dropout
        )
        self.dropout = nn.Dropout(args.dropout)

        # Convolution module: using depthwise conv followed by pointwise conv
        self.conv_module = nn.Sequential(
            nn.LayerNorm(d_model),
            # For Conv1d, input shape: (batch, channels, seq_len). We'll adjust dimensions accordingly.
            nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            ),  # depthwise conv
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

        # Second feed forward module
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        # First feed forward module
        y = self.ffn1(x)
        x = x + 0.5 * y

        # Multi-head self-attention
        y, _ = self.mha(x, x, x)
        x = x + self.dropout(y)

        # Convolution module
        # Permute x to (batch, d_model, seq_len) for Conv1d
        y = x.transpose(0, 1).transpose(1, 2)  # (batch, d_model, seq_len)
        y = self.conv_module(y)
        # Permute back to (seq_len, batch, d_model)
        y = y.transpose(1, 2).transpose(0, 1)
        x = x + self.dropout(y)

        # Second feed forward module
        y = self.ffn2(x)
        x = x + 0.5 * y

        x = self.final_layer_norm(x)
        return x


class ConformerNNet(nn.Module):
    """
    A Conformer-based neural network for the TSP problem.
    This network embeds 2D node coordinates, processes them with several Conformer blocks,
    and outputs a policy (per-node scores) and a value (scalar per instance).
    """

    def __init__(self, game, args):
        super(ConformerNNet, self).__init__()
        self.args = args
        self.num_layers = args.num_layers

        # Input layer: project 2D coordinates into an embedding space
        self.embed_dim = args.embedding_dim
        self.input_layer = nn.Linear(2, self.embed_dim)

        # Stack of Conformer blocks
        self.layers = nn.ModuleList(
            [ConformerBlock(self.embed_dim, args) for _ in range(self.num_layers)]
        )

        # Policy head: outputs a score for each node
        self.policy_head = nn.Linear(self.embed_dim, 1)

        # Value head: global average pooling followed by a few linear layers to output a scalar value
        self.value_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x, adj=None):
        # x: (batch, num_nodes, 2) where each node is a 2D coordinate; 'adj' is ignored for Conformer
        batch_size, num_nodes, _ = x.shape

        # Flatten to shape (batch*num_nodes, 2) and apply input layer
        out = self.input_layer(
            x.view(-1, 2)
        )  # (batch*num_nodes, D), where D is output dim
        embed_dim = out.size(1)
        x = out.view(batch_size, num_nodes, embed_dim)  # (batch, num_nodes, D)

        # Prepare for Conformer blocks: transpose to (num_nodes, batch, embed_dim)
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x)

        # Transpose back to (batch, num_nodes, embed_dim)
        x = x.transpose(0, 1)

        # Policy head: compute per-node scores
        policy = self.policy_head(x).squeeze(-1)  # (batch, num_nodes)

        # Value head: compute scalar value with global average pooling
        value = self.value_head(x.mean(dim=1)).squeeze(-1)  # (batch,)

        return policy, value


# Simple test to verify the network works
if __name__ == "__main__":

    class Args:
        num_layers = 2
        heads = 4
        dropout = 0.1
        embedding_dim = 128

    dummy_args = Args()
    net = ConformerNNet(None, dummy_args)
    x = torch.randn(32, 10, 2)  # batch of 32, 10 nodes, 2 coordinates per node
    policy, value = net(x)
    print("Policy shape:", policy.shape)  # Expected: (32, 10)
    print("Value shape:", value.shape)  # Expected: (32,)
