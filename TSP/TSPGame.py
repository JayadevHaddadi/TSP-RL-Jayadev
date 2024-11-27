from __future__ import print_function
import sys

sys.path.append("..")
from .TSPState import TSPState
import numpy as np


class TSPGame:

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
        b = TSPState(self.num_nodes, self.node_coordinates)
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
    
    def getNextState(self, board, action):
        """
        Given the current state and action, return the next state.

        Args:
            board (np.array): Current tour.
            action (int): Action to apply (index of the possible action).

        Returns:
            next_board (np.array): The new tour after applying the action.
        """
        b = TSPState(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        b.execute_action(action)
        return b.get_state()

    def getValidMoves(self, board):
        """
        Return a binary vector of valid moves.

        For TSP, all 2-opt moves are usually valid.
        """
        b = TSPState(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        valid_moves = b.get_valid_actions()
        return np.array(valid_moves)

    def getGameEnded(self, board):
        """
        Return the game result.

        Since TSP is an optimization problem, we can define a stopping criterion,
        such as reaching a maximum number of steps or no improvement.

        Returns:
            result (float): 0 if game is not ended, otherwise the negative tour length.
        """
        b = TSPState(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        if b.is_terminal():
            # Return the negative tour length as the reward to be maximized
            return -b.get_tour_length()
        else:
            return 0

    def getCanonicalForm(self, board):
        """
        Return the canonical form of the board.

        For TSP, the canonical form can be the tour starting from a fixed node.
        """
        b = TSPState(self.num_nodes, self.node_coordinates)
        b.set_state(np.copy(board))
        canonical_tour = b.get_canonical_tour()
        return canonical_tour

    def getSymmetries(self, board, pi):
        """
        Return symmetric versions of the board and policy vector by rearranging
        node order and adjusting the tour accordingly.
        """
        symmetries = []

        # Original board and pi
        symmetries.append((board, pi))

        # Generate permutations of node indices
        # For example, swap pairs of nodes
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                # Create a copy of node coordinates with nodes i and j swapped
                new_node_coords = self.node_coordinates.copy()
                new_node_coords[i], new_node_coords[j] = new_node_coords[j], new_node_coords[i]

                # Adjust the tour order accordingly
                new_board = board.copy()
                new_board = [j if x == i else i if x == j else x for x in new_board]

                # Adjust the policy vector accordingly
                new_pi = pi.copy()
                # You may need to adjust the indices in pi to match the new node order
                # This depends on how actions are defined in your implementation

                # Append the new symmetry
                symmetries.append((new_board, new_pi))

        return symmetries

    def getTourLength(self, board):
        """
        Computes the tour length for the given board (tour).
        """
        length = 0
        for i in range(len(board)):
            from_node = board[i]
            to_node = board[(i + 1) % len(board)]
            x1, y1 = self.node_coordinates[from_node]
            x2, y2 = self.node_coordinates[to_node]
            dist = np.hypot(x2 - x1, y2 - y1)
            length += dist
        return length

    def stringRepresentation(self, board):
        # print(board)
        # print(type(board))
        # flat_board = [item for sublist in board for item in sublist]
        # return ",".join(map(str, flat_board))
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
