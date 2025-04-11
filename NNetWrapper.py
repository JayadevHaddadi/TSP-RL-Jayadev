import os
import sys
import logging

import numpy as np
from tqdm import tqdm


sys.path.append("../../")
from TSPGame import TSPGame
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


###################################
# NNetWrapper Class
###################################
class NNetWrapper(NeuralNet):
    def __init__(self, game: TSPGame, args):  # args is now a dictionary
        # Change all args.architecture to args['architecture']
        if args.get("architecture", "default") == "pointer":
            from TSPNNet_Pointer import TSPNNet_Pointer

            self.nnet = TSPNNet_Pointer(game, args)
        elif args.get("architecture") == "transformer_deepseek":
            from TSPNNet_Transformer_deepseek import TransformerModel

            self.nnet = TransformerModel(game, args)  # Fixed class name
        elif args.get("architecture") == "gcn":
            from TSPNNet_GCN import TSPNNet

            self.nnet = TSPNNet(game, args)
        elif args.get("architecture") == "conformer":
            from TSPNNet_Conformer import ConformerNNet

            self.nnet = ConformerNNet(game, args)
        elif args.get("architecture") == "graphpointer":
            from TSPNNet_GraphPointer import GraphPointerNet

            self.nnet = GraphPointerNet(game, args)
        else:
            raise Exception("No known NN architecture")

        self.game = game
        self.args = args
        self.board_size = game.getNumberOfNodes()
        self.action_size = game.getActionSize()
        self.node_coordinates = np.array(game.node_coordinates)[: self.board_size]
        coords_min = self.node_coordinates.min(axis=0)
        coords_max = self.node_coordinates.max(axis=0)
        self.normalized_coords = (self.node_coordinates - coords_min) / (
            coords_max - coords_min + 1e-8
        )
        # Convert to tensor and move to device upfront
        self.normalized_coords = torch.FloatTensor(self.normalized_coords)

        if self.args.cuda:
            self.normalized_coords = self.normalized_coords.cuda()
            self.nnet.cuda()

        # Modify the policy head initialization
        self.fc_pi = nn.Linear(args.num_channels, self.action_size)
        nn.init.kaiming_normal_(self.fc_pi.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc_pi.bias, 0.0)
        self.print_model_info()

    def prepare_input(self, state):
        if self.args.get("architecture") == "conformer":
            # For conformer, ignore state-specific info and use raw 2D coordinates
            nf = self.normalized_coords.unsqueeze(0)  # shape: (1, num_nodes, 2)
            dummy_adj = torch.zeros(
                (1, self.board_size, self.board_size),
                device=self.normalized_coords.device,
            )
            return nf, dummy_adj
        num_nodes = self.board_size
        normalized_coords = self.normalized_coords  # Use precomputed tensor

        tour = state.tour
        tour_positions = torch.zeros(num_nodes, device=normalized_coords.device)
        is_visited = torch.zeros(num_nodes, device=normalized_coords.device)
        for idx, node in enumerate(tour):
            if num_nodes > 1:
                tour_positions[node] = idx / (num_nodes - 1)
            else:
                tour_positions[node] = 0.0
            is_visited[node] = 1

        node_features = torch.cat(
            (normalized_coords, tour_positions.unsqueeze(1), is_visited.unsqueeze(1)),
            dim=1,
        )

        # Compute adjacency matrix as a tensor directly
        adjacency_matrix = torch.zeros(
            (num_nodes, num_nodes), device=normalized_coords.device
        )
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            adjacency_matrix[from_node, to_node] = 1
            adjacency_matrix[to_node, from_node] = 1

        # print("nodeFeatures\n" , node_features.unsqueeze(0))
        # print("Adjmatrix\n",adjacency_matrix.unsqueeze(0))

        return node_features.unsqueeze(0), adjacency_matrix.unsqueeze(0)

    def train(self, examples):
        # Check if the underlying nnet has the specific parameter methods
        if (
            hasattr(self.nnet, "policy_parameters")
            and hasattr(self.nnet, "value_parameters")
            and hasattr(self.nnet, "shared_parameters")
        ):
            print("Using specific parameter groups for optimizer.")
            optimizer = optim.Adam(
                [
                    {"params": self.nnet.policy_parameters(), "lr": self.args.pi_lr},
                    {
                        "params": self.nnet.value_parameters(),
                        "lr": self.args.learning_rate,
                    },  # Assuming value uses main LR
                    {
                        "params": self.nnet.shared_parameters(),
                        "lr": self.args.learning_rate,
                    },  # Assuming shared uses main LR
                ]
            )
        else:
            # Fallback to the less reliable name-based method or just a single LR
            print(
                "Warning: Using fallback optimizer setup (single LR or name-based). Name-based might be incorrect for GCN."
            )
            # Option 1: Fallback to single LR for all params (safer than wrong name-based)
            optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.learning_rate)
            # Option 2: Keep the potentially incorrect name-based logic as a fallback (use with caution)
            # policy_params = []
            # other_params = []
            # for name, param in self.nnet.named_parameters():
            #     if "pi" in name: policy_params.append(param)
            #     else: other_params.append(param)
            # optimizer = optim.Adam([
            #     {'params': policy_params, 'lr': self.args.pi_lr},
            #     {'params': other_params, 'lr': self.args.learning_rate},
            # ])

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_decay
        )

        pi_loss_list = []
        v_loss_list = []

        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            np.random.shuffle(examples)
            batch_count = int(len(examples) / self.args.batch_size)
            if batch_count == 0:
                batch_count = 1

            batches = tqdm(range(batch_count), desc="Training Net")
            for batch_idx in batches:
                sample_ids = np.arange(
                    batch_idx * self.args.batch_size,
                    min((batch_idx + 1) * self.args.batch_size, len(examples)),
                )
                batch_examples = [examples[j] for j in sample_ids]

                # Each example is (partial_state, target_pi, leftover_distance)
                states, target_pis, target_vs = zip(*batch_examples)

                node_features_list = []
                adjacency_list = []
                for st in states:
                    nf, adj = self.prepare_input(st)
                    node_features_list.append(nf)
                    adjacency_list.append(adj)

                node_features = torch.cat(node_features_list, dim=0)
                adjacency = torch.cat(adjacency_list, dim=0)

                target_pis = torch.FloatTensor(np.array(target_pis))
                target_vs = torch.FloatTensor(np.array(target_vs).astype(np.float32))

                if self.args.cuda:
                    target_pis, target_vs = target_pis.cuda(), target_vs.cuda()

                out_pi, out_v = self.nnet(node_features, adjacency)
                # out_v is leftover distance the network predicts

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(
                    target_vs, out_v
                )  # MSE between leftover_distance and predicted leftover
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), node_features.size(0))
                v_losses.update(l_v.item(), node_features.size(0))
                batches.set_postfix(Loss_pi=pi_losses.avg, Loss_v=v_losses.avg)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.nnet.parameters(), self.args.max_gradient_norm
                )
                optimizer.step()

            pi_loss_list.append(pi_losses.avg)
            v_loss_list.append(v_losses.avg)
            scheduler.step()
            print("Current LR:", scheduler.get_last_lr())

        return pi_loss_list[-1], v_loss_list[-1]

    def predict(self, state):
        self.nnet.eval()
        with torch.no_grad():
            nf, adj = self.prepare_input(state)
            raw_logits, out_v = self.nnet(nf, adj)
            # Show actual raw logits before any transformations
            # logging.info(f"Raw logits: {raw_logits.cpu().numpy()[0]}")
            # Apply softmax for probability conversion
            pi = (
                F.softmax(raw_logits / 2.0, dim=1).cpu().numpy()[0]
            )  # Explicit temperature
            v = out_v.cpu().item()
            return pi, v

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.nnet.state_dict(), filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = torch.device("cuda" if self.args.cuda else "cpu")
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint)

    def loss_pi(self, targets, outputs):
        # Fix numerical stability by using log_softmax directly
        log_probs = F.log_softmax(outputs, dim=1)
        loss = -torch.sum(targets * log_probs) / targets.size(0)
        return loss

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)
        return loss

    def forward(self, x):
        pi = self.fc_pi(x)
        pi = torch.clamp(pi, min=-50, max=50)  # Prevent extreme values
        pi = pi / 2.0  # Temperature scaling
        return pi, F.tanh(self.fc_v(x))

    def print_model_info(self):
        """Prints the full architecture of the neural network and its total parameter count."""

        total_params = sum(p.numel() for p in self.nnet.parameters())
        logging.info("Neural network architecture:\n%s", str(self.nnet))
        logging.info("Total parameter count: %d", total_params)
