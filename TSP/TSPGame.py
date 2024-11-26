from __future__ import print_function
import sys

sys.path.append("..")
from Game import Game
from .TSPLogic import TSPBoard
import numpy as np


class TSPGame(Game):

    def __init__(self, num_nodes, node_coordinates):
        """
        Initialize the TSP game.

        Args:
            num_nodes (int): Number of nodes in the TSP.
            node_coordinates (list of tuples): Coordinates of the nodes.
        """
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates

    def getInitBoard(self):
        """
        Return the initial board state.

        For TSP, this could be a random tour or a fixed starting point.
        """
        b = TSPBoard(self.num_nodes, self.node_coordinates)
        return b.get_initial_state()

    def getBoardSize(self):
        """
        Return the size of the board.

        This depends on how we represent the TSP state.
        For simplicity, we'll represent the tour as a sequence of node indices.
        """
        # Return the length of the tour representation
        return (self.num_nodes,)

    def getActionSize(self):
        """
        Return the total number of possible actions.

        For TSP, actions could be 2-opt swaps between any pair of edges.
        The number of possible 2-opt moves is (n*(n-1))/2
        """
        return (self.num_nodes * (self.num_nodes - 1)) // 2

    def getNextState(self, board, player, action):
        """
        Given the current state, player, and action, return the next state and next player.

        Args:
            board (np.array): Current tour.
            player (int): Current player (always 1 for TSP).
            action (int): Action to apply (index of the possible action).

        Returns:
            (next_board, next_player)
        """
        b = TSPBoard(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        b.execute_action(action)
        # For TSP, player remains the same
        return (b.get_state(), player)

    def getValidMoves(self, board, player):
        """
        Return a binary vector of valid moves.

        For TSP, all 2-opt moves are usually valid.
        """
        b = TSPBoard(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        valid_moves = b.get_valid_actions()
        return np.array(valid_moves)

    def getGameEnded(self, board, player):
        """
        Return the game result.

        Since TSP is an optimization problem, we can define a stopping criterion,
        such as reaching a maximum number of steps or no improvement.

        Returns:
            result (float): 0 if game is not ended, otherwise the negative tour length.
        """
        b = TSPBoard(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        if b.is_terminal():
            # Return the negative tour length as the reward to be maximized
            return -b.get_tour_length()
        else:
            return 0

    def getCanonicalForm(self, board, player):
        """
        Return the canonical form of the board.

        For TSP, the canonical form can be the tour starting from a fixed node.
        """
        b = TSPBoard(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        canonical_tour = b.get_canonical_tour()
        return canonical_tour

    def getSymmetries(self, board, pi):
        """
        Return symmetric versions of the board and policy vector.

        For TSP, symmetries could include reversing the tour.
        """
        symmetries = []
        b = TSPBoard(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        # Original
        symmetries.append((board, pi))
        # Reversed tour
        reversed_board = board[::-1]
        reversed_pi = pi[::-1]
        symmetries.append((reversed_board, reversed_pi))
        return symmetries

    def stringRepresentation(self, board):
        """
        Return a string representation of the board.
        """
        return ",".join(map(str, board))

    @staticmethod
    def display(board):
        """
        Display the tour and its length.
        """
        print("Tour:", board)
        # Assuming we have access to the node coordinates
        # Compute the tour length
        length = 0
        for i in range(len(board)):
            from_node = board[i]
            to_node = board[(i + 1) % len(board)]
            x1, y1 = self.node_coordinates[from_node]
            x2, y2 = self.node_coordinates[to_node]
            dist = np.hypot(x2 - x1, y2 - y1)
            length += dist
        print("Tour Length:", length)
