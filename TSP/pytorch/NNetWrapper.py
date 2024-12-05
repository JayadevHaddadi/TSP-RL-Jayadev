import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("../../")
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .TSPNNet import TSPNNet as nnet
import csv


class NNetWrapper(NeuralNet):
    def __init__(self, game, args):
        self.nnet = nnet(game, args)
        self.game = game
        self.args = args
        self.board_size = game.getBoardSize()[0]  # Number of nodes
        self.action_size = game.getActionSize()
        self.node_coordinates = np.array(game.node_coordinates)  # Shape: (num_nodes, 2)

        if self.args.cuda:
            self.nnet.cuda()

    def prepare_input(self, tsp_state):
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
        normalized_coords = (node_coords - coords_min) / (
            coords_max - coords_min + 1e-8
        )

        # Get tour positions and normalize to [0, 1]
        tour_positions = np.zeros(num_nodes)
        for idx, node in enumerate(tsp_state.tour):
            tour_positions[node] = idx / (num_nodes - 1)

        # Combine features: x, y, position
        node_features = np.hstack(
            (normalized_coords, tour_positions.reshape(-1, 1))
        )  # Shape: (num_nodes, 3)

        # Create adjacency matrix based on the current tour
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            from_node = tsp_state.tour[i]
            to_node = tsp_state.tour[(i + 1) % num_nodes]
            adjacency_matrix[from_node, to_node] = 1
            adjacency_matrix[to_node, from_node] = 1  # Since the graph is undirected

        # Convert to tensors
        node_features = torch.FloatTensor(node_features)
        adjacency_matrix = torch.FloatTensor(adjacency_matrix)

        if self.args.cuda:
            node_features = node_features.cuda()
            adjacency_matrix = adjacency_matrix.cuda()

        return node_features, adjacency_matrix

    def train(self, examples, iteration=0, losses_file=None):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)

        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            # Shuffle examples before training
            np.random.shuffle(examples)

            batch_count = int(len(examples) / self.args.batch_size)
            if batch_count == 0:
                batch_count = 1  # Ensure at least one batch

            batches = tqdm(range(batch_count), desc="Training Net")
            for batch_idx in batches:
                sample_ids = np.arange(
                    batch_idx * self.args.batch_size,
                    min((batch_idx + 1) * self.args.batch_size, len(examples)),
                )
                batch_examples = [examples[j] for j in sample_ids]

                tsp_state, target_pis, target_vs = zip(*batch_examples)

                # Prepare input features and adjacency matrices
                node_features_list = []
                adjacency_matrices_list = []
                for board in tsp_state:
                    node_features, adjacency_matrix = self.prepare_input(board)
                    node_features_list.append(node_features)
                    adjacency_matrices_list.append(adjacency_matrix)

                # Stack inputs to create batch tensors
                node_features = torch.stack(
                    node_features_list
                )
                adjacency_matrices = torch.stack(
                    adjacency_matrices_list
                )

                target_pis = torch.FloatTensor(np.array(target_pis))
                target_vs = torch.FloatTensor(np.array(target_vs).astype(np.float32))

                if self.args.cuda:
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
                batches.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)

                # Append losses to CSV file
                if losses_file:
                    with open(losses_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([iteration, epoch+1, batch_idx+1, l_pi.item(), l_v.item()])

                # Backpropagation and optimization step
                optimizer.zero_grad()
                total_loss.backward()
                # Optional: Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.nnet.parameters(), self.args.max_gradient_norm
                )
                optimizer.step()

        # No need to return losses


    def predict(self, tsp_state):
        """
        Predicts the policy and value for the given tsp_state.

        Args:
            tsp_state: TSPState instance representing the current state.

        Returns:
            pi: numpy array of shape [action_size], policy probabilities
            v: float, value estimate
        """
        # Prepare input
        node_features, adjacency_matrix = self.prepare_input(tsp_state)

        # Add batch dimension
        node_features = node_features.unsqueeze(
            0
        )  # Shape: [1, num_nodes, node_feature_size]
        adjacency_matrix = adjacency_matrix.unsqueeze(
            0
        )  # Shape: [1, num_nodes, num_nodes]

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(node_features, adjacency_matrix)

        # Convert outputs to numpy arrays
        pi = pi.cpu().numpy()[0]  # Shape: [action_size]
        v = v.cpu().item()

        return pi, v

    def loss_pi(self, targets, outputs):
        """
        Calculates the policy loss.

        Args:
            targets: tensor of shape [batch_size, action_size], target policy probabilities
            outputs: tensor of shape [batch_size, action_size], predicted policy probabilities

        Returns:
            loss: scalar tensor representing the policy loss
        """
        # Use cross-entropy loss
        loss = -torch.sum(targets * torch.log(outputs + 1e-8)) / targets.size(0)
        return loss

    def loss_v(self, targets, outputs):
        """
        Calculates the value loss.

        Args:
            targets: tensor of shape [batch_size], target values
            outputs: tensor of shape [batch_size, 1], predicted values

        Returns:
            loss: scalar tensor representing the value loss
        """
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)
        return loss

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.nnet.state_dict(), filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = torch.device("cuda" if self.args.cuda else "cpu")
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint)
