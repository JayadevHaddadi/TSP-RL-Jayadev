import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import random

sys.path.append("..")
from .TSPState import TSPState

import logging

log = logging.getLogger(__name__)

class TSPGame:
    def __init__(self, num_nodes, node_coordinates, args=None):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.node_type = None
        self.args = args

    def getInitState(self, randomize_start=False):
        if randomize_start:
            start = random.randint(0, self.num_nodes - 1)
            s = TSPState(self.num_nodes, self.node_coordinates)
            s.tour = [start]
            s.unvisited = np.ones(self.num_nodes, dtype=int)
            s.unvisited[start] = 0
            s.current_length = 0.0
            return s
        else:
            return TSPState(self.num_nodes, self.node_coordinates)
        
    def getInitEvalState(self, start_node):
        """
        Return a TSPState that starts from the given 'start_node'
        for consistent evaluation across old and new networks.
        """
        from .TSPState import TSPState
        s = TSPState(self.num_nodes, self.node_coordinates)
        s.tour = [start_node]
        s.unvisited = np.ones(self.num_nodes, dtype=int)
        s.unvisited[start_node] = 0
        s.current_length = 0.0
        return s

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
