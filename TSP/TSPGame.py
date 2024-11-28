import sys
import copy
import numpy as np

sys.path.append("..")
from .TSPState import TSPState


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

        # Precalculate the valid moves vector (all moves are valid)
        self.valid_moves = np.ones(self.getActionSize(), dtype=int)

    def getInitBoard(self):
        """
        Return the initial TSP state.

        For TSP, this could be a random tour or a fixed starting point.
        """
        tsp_state = TSPState(self.num_nodes, self.node_coordinates)
        tsp_state.reset_initial_state()
        return tsp_state

    def getBoardSize(self):
        """
        Return the size of the board.

        For TSP, we'll represent the tour as a sequence of node indices.
        """
        return (self.num_nodes,)

    def getActionSize(self):
        """
        Return the total number of possible actions.

        For TSP, actions are 2-opt swaps between any pair of edges.
        The number of possible 2-opt moves is (n * (n - 1)) // 2
        """
        return (self.num_nodes * (self.num_nodes - 1)) // 2

    def getNextState(self, tsp_state, action):
        """
        Given the current state and action, return the next state.
        """
        # Create a deep copy of the current state
        next_state = copy.deepcopy(tsp_state)
        # Execute the action on the copied state
        next_state.execute_action(action)
        # Convert to canonical form
        canonical_tour = next_state.get_canonical_tour()
        next_state.set_state(canonical_tour)
        return next_state

    def getValidMoves(self, tsp_state):
        """
        Return a binary vector of valid moves.

        For TSP, all 2-opt moves are valid.
        """
        return self.valid_moves

    def getCanonicalForm(self, tsp_state):
        """
        Return the canonical form of the state.

        For TSP, the canonical form can be the tour starting from a fixed node.
        """
        canonical_state = copy.deepcopy(tsp_state)
        canonical_tour = canonical_state.get_canonical_tour()
        canonical_state.set_state(canonical_tour)
        return canonical_state

    def getSymmetries(self, tsp_state, pi):
        """
        Return symmetric versions of the state and policy vector.
        """
        symmetries = []

        # Original state and pi
        symmetries.append((tsp_state, pi))

        # Reversed tour
        reversed_state = copy.deepcopy(tsp_state)
        reversed_tour = list(reversed(tsp_state.tour))
        reversed_state.set_state(reversed_tour)

        # Adjust the policy accordingly
        reversed_pi = self.adjust_policy_for_reversal(pi)

        symmetries.append((reversed_state, reversed_pi))

        # Rotated tours
        for shift in range(1, self.num_nodes):
            rotated_state = copy.deepcopy(tsp_state)
            rotated_tour = tsp_state.tour[shift:] + tsp_state.tour[:shift]
            rotated_state.set_state(rotated_tour)

            # Adjust the policy accordingly
            rotated_pi = self.adjust_policy_for_rotation(pi, shift)

            symmetries.append((rotated_state, rotated_pi))

        return symmetries

    def adjust_policy_for_rotation(self, pi, shift):
        """
        Adjusts the policy vector when the tour is rotated.
        """
        action_size = self.getActionSize()
        rotated_pi = np.zeros_like(pi)
        for action_index in range(action_size):
            i, j = self.action_index_to_edges(action_index)
            # Adjust indices based on rotation
            i_rot = (i - shift) % self.num_nodes
            j_rot = (j - shift) % self.num_nodes
            # Ensure i_rot < j_rot
            if i_rot > j_rot:
                i_rot, j_rot = j_rot, i_rot
            # Get the new action index
            new_action_index = self.edges_to_action_index(i_rot, j_rot)
            rotated_pi[new_action_index] = pi[action_index]
        return rotated_pi

    def adjust_policy_for_reversal(self, pi):
        """
        Adjusts the policy vector when the tour is reversed.
        """
        action_size = self.getActionSize()
        reversed_pi = np.zeros_like(pi)
        for action_index in range(action_size):
            i, j = self.action_index_to_edges(action_index)
            # Map the edges (i, j) to the reversed indices
            i_rev = (self.num_nodes - 1) - i
            j_rev = (self.num_nodes - 1) - j
            # Ensure i_rev < j_rev
            if i_rev > j_rev:
                i_rev, j_rev = j_rev, i_rev
            # Get the new action index
            new_action_index = self.edges_to_action_index(i_rev, j_rev)
            reversed_pi[new_action_index] = pi[action_index]
        return reversed_pi

    def getTourLength(self, tsp_state):
        """
        Computes the tour length for the given state.
        """
        return tsp_state.get_tour_length()

    def uniqueStringRepresentation(self, tsp_state):
        """
        Returns a unique string representation of the state.

        Args:
            tsp_state (TSPState): The current TSP state.

        Returns:
            str: A unique string representation of the tour.
        """
        return ",".join(map(str, tsp_state.tour))

    def display(self, tsp_state):
        """
        Display the tour and its length.

        Args:
            tsp_state (TSPState): The current TSP state.
        """
        print("Tour:", tsp_state.tour)
        # Compute and display the tour length
        length = tsp_state.get_tour_length()
        print("Tour Length:", length)

    def action_index_to_edges(self, action_index):
        """
        Convert an action index to a pair of node indices (i, j) for a 2-opt swap.
        """
        n_nodes = self.num_nodes
        total = 0
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                if total == action_index:
                    return i, j
                total += 1
        raise ValueError("Invalid action index")

    def edges_to_action_index(self, i, j):
        """
        Convert a pair of node indices (i, j) to an action index for a 2-opt swap.
        """
        n_nodes = self.num_nodes
        total = 0
        for m in range(n_nodes - 1):
            for k in range(m + 1, n_nodes):
                if (m, k) == (i, j):
                    return total
                total += 1
        raise ValueError("Invalid edge indices")
