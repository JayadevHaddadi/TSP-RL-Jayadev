import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from TSPGame import TSPGame  # Assuming TSPGame might be needed for dimensions


# Helper for GAT layer
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha  # LeakyReLU negative slope
        self.concat = concat

        # Learnable weights for linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # Learnable weights for attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: input node features [Batch, N, in_features]
        # adj: adjacency matrix [Batch, N, N] (optional, can use dense attention)
        Wh = torch.matmul(h, self.W)  # [Batch, N, out_features]
        N = Wh.size()[1]  # Number of nodes

        # Prepare attention mechanism inputs
        a_input = torch.cat(
            [Wh.repeat(1, 1, N).view(-1, N * N, self.out_features), Wh.repeat(1, N, 1)],
            dim=2,
        ).view(-1, N, N, 2 * self.out_features)
        # Calculate attention scores (e)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [Batch, N, N]

        # Masked attention (optional: only attend to neighbors if adj is provided)
        zero_vec = -9e15 * torch.ones_like(e)
        # Use adjacency matrix if provided and sparse, otherwise use dense attention
        # For TSP, dense attention often works well. Let's use dense here.
        # if adj is not None:
        #     attention = torch.where(adj > 0, e, zero_vec)
        # else:
        attention = e  # Dense attention

        attention = F.softmax(attention, dim=2)  # [Batch, N, N]
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # [Batch, N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# Encoder using GAT layers
class GraphPointerEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, dropout, alpha):
        super(GraphPointerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.attentions = nn.ModuleList()
        for _ in range(num_layers):
            layer_attentions = [
                GraphAttentionLayer(
                    embedding_dim,
                    embedding_dim // num_heads,
                    dropout=dropout,
                    alpha=alpha,
                    concat=True,
                )
                for _ in range(num_heads)
            ]
            self.attentions.append(nn.ModuleList(layer_attentions))
        # Activation for concatenated heads
        self.elu = nn.ELU()

    def forward(self, node_features, adj=None):
        x = self.embedding(node_features)
        x = F.dropout(x, p=0.1, training=self.training)  # Initial dropout

        for i, layer in enumerate(self.attentions):
            # Concatenate heads
            x_heads = torch.cat([att(x, adj) for att in layer], dim=-1)
            x = self.elu(x_heads)  # Apply activation after concat
            if i < len(self.attentions) - 1:  # Apply dropout between layers
                x = F.dropout(x, p=0.1, training=self.training)

        return x  # Final node embeddings [Batch, N, embedding_dim]


# Decoder using Pointer Network mechanism
class GraphPointerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(GraphPointerDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0
        self.head_dim = embedding_dim // num_heads

        # Query, Key, Value projections for attention
        self.W_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # No Value projection needed for pointer, we use nodes directly

        # Used for calculating compatibility score
        self.v = nn.Parameter(torch.Tensor(embedding_dim))
        self.v.data.uniform_(
            -1.0 / math.sqrt(embedding_dim), 1.0 / math.sqrt(embedding_dim)
        )

    def forward(self, encoder_output, context_embedding, mask):
        # encoder_output: [Batch, N, embedding_dim] (Node embeddings from encoder)
        # context_embedding: [Batch, embedding_dim] (e.g., embedding of current node + graph embedding)
        # mask: [Batch, N] (Mask for already visited nodes, 1 = valid, 0 = invalid)

        batch_size, seq_len, _ = encoder_output.size()

        # Project context to query and encoder outputs to keys
        query = self.W_query(context_embedding).unsqueeze(
            1
        )  # [Batch, 1, embedding_dim]
        keys = self.W_key(encoder_output)  # [Batch, N, embedding_dim]

        # --- Calculate Attention Scores (Compatibility) ---
        # Simple additive attention variant for Pointer Networks
        features = torch.tanh(query + keys)  # [Batch, N, embedding_dim]
        # Project features down to a single score per node
        scores = torch.matmul(features, self.v)  # [Batch, N]

        # Apply mask: set scores of masked elements to -inf before softmax
        scores_masked = scores.masked_fill(mask == 0, -float("inf"))

        # --- Policy Output (Logits) ---
        # The masked scores *are* the logits for the next action
        policy_logits = scores_masked

        return policy_logits


# Value Head
class ValueHead(nn.Module):
    def __init__(self, embedding_dim):
        super(ValueHead, self).__init__()
        self.layer1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.layer2 = nn.Linear(embedding_dim // 2, 1)

    def forward(self, graph_embedding):
        # graph_embedding: [Batch, embedding_dim]
        x = F.relu(self.layer1(graph_embedding))
        value = self.layer2(x)
        return value  # Estimated remaining cost/negative length


# Main Graph Pointer Network Class
class GraphPointerNet(nn.Module):
    def __init__(self, game: TSPGame, args):
        super(GraphPointerNet, self).__init__()
        self.args = args
        self.board_size = game.getNumberOfNodes()
        self.input_dim = 4  # x, y, tour_pos, visited (adjust if different)

        # Components
        self.encoder = GraphPointerEncoder(
            input_dim=self.input_dim,
            embedding_dim=args.embedding_dim,
            num_heads=args.heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            alpha=0.2,  # LeakyReLU alpha for GAT
        )
        self.decoder = GraphPointerDecoder(
            embedding_dim=args.embedding_dim, num_heads=args.heads
        )
        self.value_head = ValueHead(embedding_dim=args.embedding_dim)

    def forward(self, node_features, adj_matrix, current_node_idx=None):
        # node_features: [Batch, N, input_dim]
        # adj_matrix: [Batch, N, N] (can be None if encoder doesn't use it)
        # current_node_idx: [Batch] index of the current node in the tour (needed for context)

        # 1. Encode node features
        node_embeddings = self.encoder(
            node_features, adj_matrix
        )  # [Batch, N, embedding_dim]

        # 2. Create Decoder Context & Mask
        # Context can be graph embedding + current node embedding
        graph_embedding = node_embeddings.mean(dim=1)  # [Batch, embedding_dim]

        # Get current node embedding (need current_node_idx)
        if current_node_idx is None:
            # If no current node (start of episode), use graph embedding only or avg?
            # Using just graph embedding is simpler
            context_embedding = graph_embedding
            # Alternatively, could use a learnable start token embedding
        else:
            # Gather current node embeddings based on indices
            batch_indices = torch.arange(
                node_embeddings.size(0), device=node_embeddings.device
            )
            current_node_embedding = node_embeddings[
                batch_indices, current_node_idx
            ]  # [Batch, embedding_dim]
            context_embedding = (
                graph_embedding + current_node_embedding
            )  # Combine graph and current node info

        # Create mask from 'is_visited' feature (assuming it's the last feature)
        # visited = 1, unvisited = 0. Mask needs 1 for valid (unvisited), 0 for invalid (visited)
        is_visited_feature = node_features[:, :, -1]  # [Batch, N]
        mask = 1 - is_visited_feature  # Invert: 1 for unvisited, 0 for visited

        # 3. Decode (Pointer Attention)
        policy_logits = self.decoder(
            node_embeddings, context_embedding, mask
        )  # [Batch, N]

        # 4. Value Prediction
        value = self.value_head(graph_embedding)  # [Batch, 1]

        return policy_logits, value.squeeze(-1)  # Return logits and scalar value

    # --- Add parameter grouping methods for NNetWrapper ---
    def policy_parameters(self):
        # Parameters specific to policy generation (decoder)
        return self.decoder.parameters()

    def value_parameters(self):
        # Parameters specific to value prediction (value_head)
        return self.value_head.parameters()

    def shared_parameters(self):
        # Parameters used by both (encoder)
        decoder_ids = {id(p) for p in self.decoder.parameters()}
        value_ids = {id(p) for p in self.value_head.parameters()}
        for p in self.parameters():
            if id(p) not in decoder_ids and id(p) not in value_ids:
                yield p

    # -----------------------------------------------------
