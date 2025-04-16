import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from TSPGame import TSPGame  # Assuming TSPGame might be needed for dimensions

# --- Helper Modules ---


class Swish(nn.Module):
    """Swish activation function"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Split tensor along the last dimension
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class DepthwiseConv1d(nn.Module):
    """Depthwise Separable Convolution 1D"""

    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, groups=chan_in, padding=padding
        )

    def forward(self, x):
        return self.conv(x)


# --- Conformer Building Blocks ---


class ConformerFeedForward(nn.Module):
    """FeedForward Module for Conformer"""

    def __init__(self, dim, expansion_factor=4, dropout=0.1):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2),  # Double dim for GLU
            GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Convolution Module for Conformer"""

    def __init__(self, dim, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = kernel_size // 2  # Maintain sequence length

        # --- Separate LayerNorm ---
        self.layer_norm = nn.LayerNorm(dim)
        # -------------------------

        # --- Convolution Sequence (without initial LayerNorm) ---
        self.conv_sequence = nn.Sequential(
            # nn.LayerNorm(dim), # REMOVED FROM HERE
            nn.Conv1d(dim, inner_dim * 2, 1),  # Pointwise Conv x 2 (for GLU)
            GLU(dim=1),  # Apply GLU along the channel dimension
            DepthwiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim),  # BatchNorm usually applied after DepthwiseConv
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),  # Pointwise Conv back to original dim
            nn.Dropout(dropout),
        )
        # ------------------------------------------------------

    def forward(self, x):
        # Input x shape: [Batch, SeqLen, Dim]
        residual = x  # Store residual for skip connection

        # --- Apply LayerNorm BEFORE transpose ---
        x = self.layer_norm(x)
        # ---------------------------------------

        # Transpose for Conv1d: [Batch, SeqLen, Dim] -> [Batch, Dim, SeqLen]
        x = x.transpose(1, 2)

        # Apply convolution sequence
        x = self.conv_sequence(x)

        # Transpose back: [Batch, Dim, SeqLen] -> [Batch, SeqLen, Dim]
        x = x.transpose(1, 2)

        # Add residual - Make sure it's added correctly in the ConformerBlock forward pass
        # We'll return the processed x, the ConformerBlock handles adding residual
        return x  # Return the processed tensor for the block to add


class ConformerAttentionModule(nn.Module):
    """Multi-Head Self-Attention Module"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        # x: [Batch, SeqLen, Dim]
        b, n, _, h = *x.shape, self.num_heads

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # Split into q, k, v
        q, k, v = map(
            lambda t: t.reshape(b, n, h, self.head_dim).transpose(1, 2), qkv
        )  # [Batch, Heads, SeqLen, HeadDim]

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim**-0.5)

        # Apply mask if provided (e.g., for padding)
        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            attn_mask = (
                mask[:, None, :, None] * mask[:, None, None, :]
            )  # Create attention mask [B, 1, N, N]
            dots = dots.masked_fill(~attn_mask, mask_value)

        attn = dots.softmax(dim=-1)
        attn = F.dropout(
            attn, p=0.1, training=self.training
        )  # Dropout on attention weights

        out = torch.matmul(attn, v)  # [Batch, Heads, SeqLen, HeadDim]
        out = out.transpose(1, 2).reshape(
            b, n, -1
        )  # Concatenate heads back -> [Batch, SeqLen, Dim]
        return self.to_out(out)


# --- Conformer Block ---
class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ff_expansion_factor,
        conv_expansion_factor,
        conv_kernel_size,
        dropout,
    ):
        super().__init__()
        self.ff1 = ConformerFeedForward(dim, ff_expansion_factor, dropout)
        self.attn = ConformerAttentionModule(dim, num_heads, dropout)
        self.conv = ConformerConvModule(
            dim, conv_kernel_size, conv_expansion_factor, dropout
        )
        self.ff2 = ConformerFeedForward(dim, ff_expansion_factor, dropout)
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # Each module expects residual connection scaled by 0.5
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, mask=mask)
        # Apply conv module and add residual
        x = x + self.conv(
            x
        )  # self.conv now returns processed features, add residual here
        x = x + 0.5 * self.ff2(x)
        x = self.norm_final(x)
        return x


# --- Main Conformer Network for TSP ---
class ConformerNNet(nn.Module):
    def __init__(self, game: TSPGame, args):
        super(ConformerNNet, self).__init__()
        self.args = args
        self.board_size = game.getNumberOfNodes()
        self.action_size = game.getActionSize()
        # Conformer often works directly on coordinates + potentially learned embeddings
        # Or could use the same 4 features as GCN/GAT
        self.input_dim = args.get(
            "conformer_input_dim", 2
        )  # Default to just x, y coordinates

        self.embedding_dim = args.embedding_dim

        # Initial projection/embedding layer
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(args.dropout),
        )

        # Conformer Blocks
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dim=self.embedding_dim,
                    num_heads=args.heads,
                    ff_expansion_factor=args.get("ff_expansion_factor", 4),
                    conv_expansion_factor=args.get("conv_expansion_factor", 2),
                    conv_kernel_size=args.get("conv_kernel_size", 31),
                    dropout=args.dropout,
                )
                for _ in range(args.num_layers)
            ]
        )

        # Global pooling layer
        self.global_pool_type = args.get("global_pool", "mean")

        # Policy Head MLP
        policy_mlp_layers = []
        current_dim = self.embedding_dim
        for _ in range(max(1, args.policy_layers) - 1):
            policy_mlp_layers.extend(
                [
                    nn.Linear(current_dim, args.policy_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                ]
            )
            current_dim = args.policy_dim
        policy_mlp_layers.append(nn.Linear(current_dim, self.action_size))
        self.policy_head = nn.Sequential(*policy_mlp_layers)

        # Value Head MLP
        value_mlp_layers = []
        current_dim = self.embedding_dim
        for _ in range(max(1, args.value_layers) - 1):
            value_mlp_layers.extend(
                [
                    nn.Linear(current_dim, args.value_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                ]
            )
            current_dim = args.value_dim
        value_mlp_layers.append(nn.Linear(current_dim, 1))
        self.value_head = nn.Sequential(*value_mlp_layers)

    def forward(
        self, node_features, adj_matrix=None
    ):  # Adj_matrix is ignored by this Conformer design
        # node_features: [Batch, N, input_dim] - could be just coords or richer features

        # 1. Input Projection
        x = self.input_projection(node_features)  # [Batch, N, embedding_dim]

        # Create mask if needed (e.g., if handling variable N, but here N is fixed)
        mask = None  # Assuming fixed sequence length N

        # 2. Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask=mask)  # [Batch, N, embedding_dim]

        # 3. Global Pooling
        if self.global_pool_type == "mean":
            graph_embedding = x.mean(dim=1)  # [Batch, embedding_dim]
        elif self.global_pool_type == "sum":
            graph_embedding = x.sum(dim=1)
        elif self.global_pool_type == "max":
            graph_embedding = torch.max(x, dim=1)[0]
        else:  # Default to mean
            graph_embedding = x.mean(dim=1)

        # 4. Policy Prediction
        policy_logits = self.policy_head(graph_embedding)  # [Batch, action_size]

        # 5. Value Prediction
        value = self.value_head(graph_embedding)  # [Batch, 1]

        return policy_logits, value.squeeze(-1)  # Return logits and scalar value

    # --- Add parameter grouping methods for NNetWrapper ---
    def policy_parameters(self):
        return self.policy_head.parameters()

    def value_parameters(self):
        return self.value_head.parameters()

    def shared_parameters(self):
        policy_ids = {id(p) for p in self.policy_parameters()}
        value_ids = {id(p) for p in self.value_parameters()}
        for p in self.parameters():
            if id(p) not in policy_ids and id(p) not in value_ids:
                yield p

    # -----------------------------------------------------

    # -----------------------------------------------------
