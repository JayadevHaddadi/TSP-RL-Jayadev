import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append("../../")
from TSP.TSPGame import TSPGame
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .TSPNNet import TSPNNet

###################################
# NNetWrapper Class
###################################
class NNetWrapper(NeuralNet):
    def __init__(self, game: TSPGame, args):
        self.nnet = TSPNNet(game, args)
        self.game = game
        self.args = args
        self.board_size = game.getNumberOfNodes()
        self.action_size = game.getActionSize()
        self.node_coordinates = np.array(game.node_coordinates)

        if self.args.cuda:
            self.nnet.cuda()

    def prepare_input(self, state):
        # node_features: (num_nodes, 3) - x,y normalized + position in partial tour (0 to 1)
        # adjacency: connect visited nodes in order; no return to start until end?
        num_nodes = self.board_size
        coords = self.node_coordinates
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        normalized_coords = (coords - coords_min) / (coords_max - coords_min + 1e-8) # is this necessary at each prepare?

        # Tour positions: if length of partial tour = t, node in position k has position = k/(num_nodes-1)
        tour = state.tour
        t = len(tour)
        tour_positions = np.zeros(num_nodes)
        for idx, node in enumerate(tour):
            if num_nodes > 1:
                tour_positions[node] = idx / (num_nodes - 1)
            else:
                tour_positions[node] = 0.0

        node_features = np.hstack((normalized_coords, tour_positions.reshape(-1, 1)))

        # adjacency: edges between consecutive visited nodes only
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(t - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            adjacency_matrix[from_node, to_node] = 1
            adjacency_matrix[to_node, from_node] = 1

        node_features = torch.FloatTensor(node_features)
        adjacency_matrix = torch.FloatTensor(adjacency_matrix)

        if self.args.cuda:
            node_features = node_features.cuda()
            adjacency_matrix = adjacency_matrix.cuda()

        node_features = node_features.unsqueeze(0)  # batch=1
        adjacency_matrix = adjacency_matrix.unsqueeze(0)

        return node_features, adjacency_matrix

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)

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
                l_v = self.loss_v(target_vs, out_v)  # MSE between leftover_distance and predicted leftover
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

        return pi_loss_list[-1], v_loss_list[-1]



    def predict(self, state):
        self.nnet.eval()
        with torch.no_grad():
            nf, adj = self.prepare_input(state)
            out_pi, out_v = self.nnet(nf, adj)
            pi = out_pi.cpu().numpy()[0]
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
        loss = -torch.sum(targets * torch.log(outputs + 1e-8)) / targets.size(0)
        return loss

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)
        return loss
