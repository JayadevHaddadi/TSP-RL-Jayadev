import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
from .TSPState import TSPState

import logging
log = logging.getLogger(__name__)

###################################
# TSPGame Class
###################################
class TSPGame:
    def __init__(self, num_nodes, node_coordinates):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.node_type = None

    def getInitBoard(self):
        """
        Return initial TSP state.
        Now initial state = just node 0 visited.
        """
        s = TSPState(self.num_nodes, self.node_coordinates)
        s.reset_initial_state()
        return s

    def getBoardSize(self):
        return (self.num_nodes,)

    def getActionSize(self):
        # Actions = choosing any unvisited node
        # Max unvisited = num_nodes - 1 (since node 0 is visited)
        # In worst case when no nodes visited except 0, max actions = num_nodes - 1
        return self.num_nodes - 1

    def getNextState(self, state: TSPState, action):
        next_state = copy.deepcopy(state)
        next_state.execute_action(action)
        return next_state

    def getValidMoves(self, state: TSPState):
        """
        Binary vector of length getActionSize(), indicating which actions are valid.
        If less than max actions are available (fewer unvisited nodes),
        we set corresponding entries to 0.
        """
        valid = np.zeros(self.getActionSize(), dtype=int)
        available = state.get_available_actions()
        # available is a sorted list of unvisited nodes
        # Actions = index into this available list
        # number of actual actions = len(available)
        for i in range(len(available)):
            valid[i] = 1
        return valid

    def isTerminal(self, state: TSPState):
        return state.is_terminal()

    def getCanonicalForm(self, state: TSPState):
        # Canonical form might not be as relevant here since we always start from node 0.
        # Just return the state as is.
        return state

    def getSymmetries(self, state: TSPState, pi):
        """
        With forward-building, symmetries might not make sense.
        We can just return the original (state, pi).
        """
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
        # Plot edges
        for i in range(len(tour)-1):
            from_node = tour[i]
            to_node = tour[i+1]
            plt.plot([coords[from_node, 0], coords[to_node, 0]],
                     [coords[from_node, 1], coords[to_node, 1]], 'r-')

        # Plot nodes
        plt.plot(coords[:,0], coords[:,1], 'o', markersize=10)

        # Add node numbering
        for idx,(x,y) in enumerate(coords):
            plt.text(x,y,str(idx), fontsize=12, color='green')

        if title:
            plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()