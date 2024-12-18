import numpy as np

###################################
# TSPState Class
###################################
class TSPState:
    def __init__(self, num_nodes, node_coordinates):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.tour = [0]
        self.unvisited = np.ones(num_nodes, dtype=int)
        self.unvisited[0] = 0
        self.current_length = 0.0

    def set_state(self, tour):
        """
        Set the current partial tour state.
        tour: list of nodes representing the partial tour
        """
        self.tour = list(tour)
        visited_set = set(tour)
        self.unvisited = set(range(self.num_nodes)) - visited_set
        self.current_length = self.get_tour_length()

    def get_state(self):
        """
        Return the current partial tour as np.array.
        """
        return np.array(self.tour)

    def execute_action(self, action):
        """
        Action = node ID directly.
        unvisited[node] = 1 means node is unvisited.
        """
        chosen_node = action
        if self.unvisited[chosen_node] != 1:
            raise ValueError(f"Chosen action {chosen_node} is not unvisited.")

        from_node = self.tour[-1]
        to_node = chosen_node
        dist = self.distance(from_node, to_node)
        self.current_length += dist
        self.tour.append(chosen_node)
        self.unvisited[chosen_node] = 0  # Mark chosen_node as visited

        # If all visited except the start, finalize the tour by closing the loop
        # Check if no unvisited remain:
        if np.sum(self.unvisited) == 0:
            start_node = self.tour[0]
            end_node = self.tour[-1]
            closing_dist = self.distance(end_node, start_node)
            self.current_length += closing_dist

    def is_terminal(self):
        # Terminal if sum of unvisited = 0 (no unvisited nodes)
        return np.sum(self.unvisited) == 0

    def get_tour_length(self):
        """
        Calculate the length of the current partial tour.
        If partial, just sum edges along the tour so far (no return to start yet).
        """
        length = 0.0
        for i in range(len(self.tour)-1):
            from_node = self.tour[i]
            to_node = self.tour[i+1]
            length += self.distance(from_node, to_node)
        return length

    def get_available_actions(self):
        """
        Available actions = choosing any of the unvisited nodes.
        """
        return sorted(list(self.unvisited))

    def distance(self, i, j):
        x1, y1 = self.node_coordinates[i]
        x2, y2 = self.node_coordinates[j]
        return np.hypot(x2 - x1, y2 - y1)