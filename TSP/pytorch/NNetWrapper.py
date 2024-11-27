import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .TSPNNet import TSPNNet as nnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'max_gradient_norm': 5.0,  # Optional: For gradient clipping
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = nnet(game, args)
        self.game = game
        self.board_size = game.getBoardSize()[0]  # Number of nodes
        self.action_size = game.getActionSize()
        self.node_coordinates = np.array(game.node_coordinates)  # Shape: (num_nodes, 2)

        if args.cuda:
            self.nnet.cuda()
            
    def prepare_input(self, board):
        """
        Prepares the input tensors for the neural network.

        Args:
            board: TSPState instance representing the current state.

        Returns:
            node_features: Tensor of shape [num_nodes, node_feature_size]
            adjacency_matrix: Tensor of shape [num_nodes, num_nodes]
        """
        num_nodes = self.board_size
        node_coords = self.node_coordinates  # Shape: (num_nodes, 2)

        # Normalize node coordinates to [0, 1]
        coords_min = node_coords.min(axis=0)
        coords_max = node_coords.max(axis=0)
        normalized_coords = (node_coords - coords_min) / (coords_max - coords_min + 1e-8)

        # Get tour positions and normalize to [0, 1]
        tour_positions = np.zeros(num_nodes)
        for idx, node in enumerate(board.tour):
            tour_positions[node] = idx / (num_nodes - 1)

        # Combine features: x, y, position
        node_features = np.hstack((
            normalized_coords,
            tour_positions.reshape(-1, 1)
        ))  # Shape: (num_nodes, 3)

        # Create adjacency matrix based on the current tour
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            from_node = board.tour[i]
            to_node = board.tour[(i + 1) % num_nodes]
            adjacency_matrix[from_node, to_node] = 1
            adjacency_matrix[to_node, from_node] = 1  # Since the graph is undirected

        # Convert to tensors
        node_features = torch.FloatTensor(node_features)
        adjacency_matrix = torch.FloatTensor(adjacency_matrix)

        if args.cuda:
            node_features = node_features.cuda()
            adjacency_matrix = adjacency_matrix.cuda()

        return node_features, adjacency_matrix

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            # Shuffle examples before training
            np.random.shuffle(examples)

            batch_count = int(len(examples) / args.batch_size)
            if batch_count == 0:
                batch_count = 1  # Ensure at least one batch

            t = tqdm(range(batch_count), desc='Training Net')
            for i in t:
                sample_ids = np.arange(i * args.batch_size, min((i + 1) * args.batch_size, len(examples)))
                batch_examples = [examples[j] for j in sample_ids]

                boards, target_pis, target_vs = zip(*batch_examples)

                # Prepare input features and adjacency matrices
                node_features_list = []
                adjacency_matrices_list = []
                for board in boards:
                    node_features, adjacency_matrix = self.prepare_input(board)
                    node_features_list.append(node_features)
                    adjacency_matrices_list.append(adjacency_matrix)

                # Stack inputs to create batch tensors
                node_features = torch.stack(node_features_list)  # Shape: [batch_size, num_nodes, node_feature_size]
                adjacency_matrices = torch.stack(adjacency_matrices_list)  # Shape: [batch_size, num_nodes, num_nodes]

                target_pis = torch.FloatTensor(np.array(target_pis))
                target_vs = torch.FloatTensor(np.array(target_vs).astype(np.float32))

                if args.cuda:
                    target_pis, target_vs = target_pis.cuda(), target_vs.cuda()

                # Compute output
                out_pi, out_v = self.nnet(node_features, adjacency_matrices)

                # Calculate losses
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # Record losses
                pi_losses.update(l_pi.item(), node_features.size(0))
                v_losses.update(l_v.item(), node_features.size(0))
                t.
