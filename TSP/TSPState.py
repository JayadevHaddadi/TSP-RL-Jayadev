import numpy as np


class TSPState:
    def __init__(self, num_nodes, node_coordinates):
        """
        Initialize the TSP board.

        Args:
            num_nodes (int): Number of nodes in the TSP.
            node_coordinates (list of tuples): Coordinates of the nodes.
        """
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates  # List of (x, y) tuples
        self.tour = []  # Current tour (list of node indices)
        self.max_no_improve = 10  # Stopping criterion (can be adjusted)
        self.no_improve_counter = 0
        self.best_length = float("inf")
        self.current_length = float("inf")
        self.reset_initial_state()

    def reset_initial_state(self):
        """
        Generate an initial tour.

        For simplicity, start with a random tour.
        """
        self.tour = list(range(self.num_nodes))
        np.random.shuffle(self.tour)
        self.tour = self.get_canonical_tour()
        self.current_length = self.get_tour_length()
        self.best_length = self.current_length

    def set_state(self, tour):
        """
        Set the current tour state.

        Args:
            tour (array-like): The current tour as a sequence of node indices.
        """
        self.tour = list(tour)
        self.current_length = self.get_tour_length()
        self.tour = self.get_canonical_tour()

    def get_state(self):
        """
        Return the current tour state.

        Returns:
            np.array: The current tour as an array of node indices.
        """
        return np.array(self.tour)

    def execute_action(self, action):
        """
        Apply an action to the current tour.

        For TSP, actions are 2-opt swaps identified by an index.

        Args:
            action (int): Index of the action.
        """
        i, j = self.action_index_to_edges(action)
        self.two_opt_swap(i, j)
        self.current_length = self.get_tour_length()
        self.tour = self.get_canonical_tour()

    def get_valid_actions(self):
        """
        Return a binary vector of valid actions.

        For TSP, all possible 2-opt moves are usually valid.

        Returns:
            list: A list of ones indicating all actions are valid.
        """
        num_actions = (self.num_nodes * (self.num_nodes - 1)) // 2
        return [1] * num_actions  # All actions are valid

    def get_tour_length(self):
        """
        Calculate the length of the current tour.

        Returns:
            float: The total length of the tour.
        """
        length = 0
        for i in range(self.num_nodes):
            from_node = self.tour[i]
            to_node = self.tour[(i + 1) % self.num_nodes]
            x1, y1 = self.node_coordinates[from_node]
            x2, y2 = self.node_coordinates[to_node]
            dist = np.hypot(x2 - x1, y2 - y1)
            length += dist
        return length

    def get_canonical_tour(self):
        """
        Return a canonical form of the tour.

        Returns:
            np.array: The canonical tour as an array of node indices.
        """
        min_index = min(self.tour)
        idx = self.tour.index(min_index)
        canonical_tour = self.tour[idx:] + self.tour[:idx]

        # Also consider the reversed tour
        reversed_tour = list(reversed(canonical_tour))
        if reversed_tour < canonical_tour:
            canonical_tour = reversed_tour

        return canonical_tour

    def action_index_to_edges(self, action_index):
        """
        Convert an action index to a pair of indices for a 2-opt swap.

        Args:
            action_index (int): Index of the action.

        Returns:
            tuple: (i, j) indices of the edges to swap.
        """
        # Map action_index to a pair (i, j) where i < j
        i = 0
        total = 0
        n = self.num_nodes
        for i in range(n - 1):
            for j in range(i + 1, n):
                if total == action_index:
                    return i, j
                total += 1
        # If action_index is out of bounds
        raise ValueError("Invalid action index")

    def two_opt_swap(self, i, j):
        """
        Perform a 2-opt swap by reversing the tour between positions i and j.

        Args:
            i (int): Start index of the swap.
            j (int): End index of the swap.
        """
        if i >= j:
            return
        self.tour[i : j + 1] = reversed(self.tour[i : j + 1])

    def get_available_actions(self):
        """
        Get a list of available actions (2-opt swaps).

        Returns:
            list of tuples: Each tuple contains indices (i, j) for possible swaps.
        """
        actions = []
        n = self.num_nodes
        for i in range(n - 1):
            for j in range(i + 1, n):
                actions.append((i, j))
        return actions

    def get_score(self):
        """
        Get the current score of the tour.

        For TSP, a lower tour length is better.

        Returns:
            float: Negative tour length to indicate that lower is better.
        """
        return -self.current_length
