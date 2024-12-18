import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
from .TSPState import TSPState

import logging

log = logging.getLogger(__name__)


class TSPGame:
    def __init__(self, num_nodes, node_coordinates, args=None):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.node_type = None
        self.args = args  # contains L_baseline

    def getInitState(self):
        return TSPState(self.num_nodes, self.node_coordinates)

    def getNumberOfNodes(self):
        return self.num_nodes

    def getNextState(self, state: TSPState, action):
        next_state = copy.deepcopy(state)
        next_state.execute_action(action)
        return next_state

    def getActionSize(self):
        # Actions correspond to node IDs: 0 to num_nodes-1
        return self.num_nodes

    def getValidMoves(self, state: TSPState):
        """
        Simply return the unvisited array.
        Since node 0 was visited at start, unvisited[0] = 0 always,
        ensuring we never pick node 0 again.
        """
        return state.unvisited

    def isTerminal(self, state: TSPState):
        return state.is_terminal()

    def getGameEnded(self, state):
        """
        Returns a value in [-1,1] if the state is terminal, else 0 if not terminal.

        If ended:
        raw_value = (L_baseline - L_final)/(L_baseline+1e-8)
        clipped = np.clip(raw_value, -1, 1)
        If clipped == 0 exactly, return 1e-12 to indicate ended state with no improvement.
        """
        if self.isTerminal(state):
            L_final = self.getTourLength(state)
            L_baseline = self.args.L_baseline
            raw_value = (L_baseline - L_final) / (L_baseline + 1e-8)
            clipped = float(np.clip(raw_value, -1, 1))

            # If ended and clipped == 0, return a tiny epsilon
            # to avoid confusion with non-ended states
            if np.isclose(clipped, 0.0):
                return 1e-12
            return clipped

        return 0

    def getCanonicalForm(self, state: TSPState):
        # No player switching, return state as is
        return state

    def getSymmetries(self, state: TSPState, pi):
        # No symmetries in forward build
        return [(state, pi)]

    def getTourLength(self, state: TSPState):
        return state.get_tour_length()

    def uniqueStringRepresentation(self, state: TSPState):
        return ",".join(map(str, state.tour))

    def display(self, state: TSPState):
        log.info(f"Tour: {state.tour}")
        log.info(f"Length: {state.get_tour_length()}")

    def plotTour(self, state: TSPState, title=None, save_path=None):
        coords = np.array(self.node_coordinates)
        tour = state.tour

        plt.figure()
        # Plot edges for the partial tour
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            plt.plot(
                [coords[from_node, 0], coords[to_node, 0]],
                [coords[from_node, 1], coords[to_node, 1]],
                "r-",
            )

        # Close the loop by connecting the last node back to the first node
        if len(tour) > 1:
            from_node = tour[-1]
            to_node = tour[0]
            plt.plot(
                [coords[from_node, 0], coords[to_node, 0]],
                [coords[from_node, 1], coords[to_node, 1]],
                "r--",
            )  # dashed line to show return

        plt.plot(coords[:, 0], coords[:, 1], "o", markersize=10)

        for idx, (x, y) in enumerate(coords):
            plt.text(x, y, str(idx), fontsize=12, color="green")

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
