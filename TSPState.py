import numpy as np


class TSPState:
    def __init__(
        self, num_nodes, node_coordinates, distance_matrix=None, start_node=None
    ):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.tour = []
        self.unvisited = np.ones(num_nodes, dtype=int)  # 1=unvisited, 0=visited

        # Initialize with start node if specified
        if start_node is not None:
            if start_node < 0 or start_node >= num_nodes:
                raise ValueError(f"Invalid start node: {start_node}")
            self.tour.append(start_node)
            self.unvisited[start_node] = 0

        self.current_length = 0.0
        self.distance_matrix = distance_matrix

    def execute_action(self, action):
        """
        Action = node ID (chosen_node).
        unvisited[node] = 1 means node is unvisited.
        """
        chosen_node = action
        if self.unvisited[chosen_node] != 1:
            raise ValueError(f"Chosen action {chosen_node} is not unvisited.")

        # Case 1: If the tour is empty, this is the very first node chosen.
        if len(self.tour) == 0:
            # No distance added since there's no 'from_node' yet.
            self.tour.append(chosen_node)
            self.unvisited[chosen_node] = 0

        # Case 2: Otherwise, we have at least one node in the tour already.
        else:
            from_node = self.tour[-1]
            to_node = chosen_node
            dist = self.distance(from_node, to_node)
            self.current_length += dist
            self.tour.append(chosen_node)
            self.unvisited[chosen_node] = 0

        # Now check how many unvisited remain.
        remaining_unvisited = np.sum(self.unvisited)

        if remaining_unvisited == 1:
            # Exactly one node remains, so automatically add it.
            last_node = np.where(self.unvisited == 1)[0][0]
            dist_2 = self.distance(self.tour[-1], last_node)
            self.current_length += dist_2
            self.tour.append(last_node)
            self.unvisited[last_node] = 0

            # Close the loop (now no unvisited remain).
            start_node = self.tour[0]
            end_node = self.tour[-1]
            closing_dist = self.distance(end_node, start_node)
            self.current_length += closing_dist

        elif remaining_unvisited == 0:
            # If code logic is correct, we should never see exactly 0 unvisited
            # immediately after an action, because we handle the last node above.
            raise Exception(
                "Should never have 0 remaining nodes after doing an action."
            )

    def is_terminal(self):
        # Terminal if sum of unvisited = 0 (no unvisited nodes)
        return np.sum(self.unvisited) == 0

    def get_tour_length(self):
        return self.current_length

    def get_available_actions(self):
        """
        Available actions = choosing any of the unvisited nodes.
        """
        return sorted(list(self.unvisited))

    def distance(self, i, j):
        """Get precomputed distance between nodes i and j"""
        # Safety check for indices
        if i < 0 or i >= self.num_nodes or j < 0 or j >= self.num_nodes:
            raise IndexError(f"Invalid node indices: {i}, {j}")

        return self.distance_matrix[i][j]
