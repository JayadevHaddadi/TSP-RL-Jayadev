import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import random

sys.path.append("..")
from TSPState import TSPState

import logging

log = logging.getLogger(__name__)


class TSPGame:
    def __init__(self, num_nodes, node_coordinates, node_type="rand", args=None):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.node_type = node_type
        self.args = args

        # Now this will use the correct distance calculation
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        """Precompute all pairwise distances"""
        n = self.num_nodes
        mat = np.zeros((n, n))

        # Log which distance calculation we're using
        logging.info(f"Computing {self.node_type} distance matrix")

        for i in range(n):
            for j in range(n):
                mat[i][j] = self._calculate_distance(i, j)

        # Log sample distances for verification
        if n >= 2:
            logging.info(f"Sample distance [0][1]: {mat[0][1]}")
            logging.info(f"Sample distance [1][0]: {mat[1][0]}")
        return mat

    def _calculate_distance(self, i, j):
        """Calculate distance between two nodes (used for matrix precomputation)"""
        if self.node_type == "tsplib":
            # GEO calculation for TSPLIB instances
            PI = 3.141592
            x1, y1 = self.node_coordinates[i]
            x2, y2 = self.node_coordinates[j]

            lat1 = x1 * PI / 180.0
            long1 = y1 * PI / 180.0
            lat2 = x2 * PI / 180.0
            long2 = y2 * PI / 180.0

            RRR = 6378.388
            q1 = np.cos(long1 - long2)
            q2 = np.cos(lat1 - lat2)
            q3 = np.cos(lat1 + lat2)
            dist = RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
            return np.round(dist)
        else:
            # Euclidean distance for random instances
            x1, y1 = self.node_coordinates[i]
            x2, y2 = self.node_coordinates[j]
            return np.hypot(x2 - x1, y2 - y1)

    def getInitState(self, start_node=None):
        if start_node is None:
            # Default to random start if not specified
            start_node = (
                0
                if self.args.get("fixed_start", True)
                else np.random.choice(self.num_nodes)
            )

        return TSPState(
            num_nodes=self.num_nodes,
            node_coordinates=self.node_coordinates,
            distance_matrix=self.distance_matrix,
            start_node=start_node,
        )

    def getNumberOfNodes(self):
        return self.num_nodes

    def getNextState(self, state: TSPState, action):
        next_state = copy.deepcopy(state)
        next_state.execute_action(action)
        return next_state

    def getActionSize(self):
        return self.num_nodes

    def getValidMoves(self, state: TSPState):
        return state.unvisited

    def isTerminal(self, state: TSPState):
        return state.is_terminal()

    def getFinalScore(self, state):
        if not self.isTerminal(state):
            raise Exception("Dont call final score unless state is terminal")
        return -self.getTourLength(state)

    def getCanonicalForm(self, state: TSPState):
        return state

    def getSymmetries(self, state: TSPState, pi):
        return [(state, pi)]

    def getTourLength(self, state: TSPState):
        return state.get_tour_length()

    def uniqueStringRepresentation(self, state: TSPState):
        """
        Return the string representation of the partial tour exactly
        as stored in state.tour, with no rotation or reversal.
        """
        tour = state.tour
        if not tour:
            return ""
        return ",".join(map(str, tour))

    def display(self, state: TSPState):
        log.info(f"Tour: {state.tour}")
        log.info(f"Length: {state.get_tour_length()}")

    def plotTour(self, state=None, title=None, save_path=None, input_tour=None):
        coords = np.array(self.node_coordinates)
        if input_tour is None:
            tour = state.tour if state else None
        else:
            tour = input_tour
        if tour is None:
            raise ValueError("No tour provided to plot")
        plt.figure()
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            plt.plot(
                [coords[from_node, 0], coords[to_node, 0]],
                [coords[from_node, 1], coords[to_node, 1]],
                "r-",
            )
        if len(tour) > 1:
            from_node = tour[-1]
            to_node = tour[0]
            plt.plot(
                [coords[from_node, 0], coords[to_node, 0]],
                [coords[from_node, 1], coords[to_node, 1]],
                "r--",
            )
        plt.plot(coords[:, 0], coords[:, 1], "o", markersize=10)
        for idx, (x, y) in enumerate(coords):
            plt.text(x, y, str(idx), fontsize=12, color="black")
        if title:
            plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
