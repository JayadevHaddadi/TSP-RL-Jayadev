import numpy as np

###################################
# TSPState Class
###################################
class TSPState:
    def __init__(self, num_nodes, node_coordinates):
        """
        Initialize the TSP partial state.
        Start with node 0 fixed as the starting node.
        """
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.reset_initial_state()

    def reset_initial_state(self):
        """
        The initial partial solution is just [0].
        Unvisited = all other nodes.
        """
        self.tour = [0]
        self.unvisited = set(range(1, self.num_nodes))
        self.current_length = 0.0  # Just one node, no edges yet.

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
        Action = index of a node in the set of unvisited nodes.
        We must map 'action' to the chosen node. Actions correspond to choosing
        one of the currently unvisited nodes based on a sorted list of them.
        """
        unvisited_list = sorted(list(self.unvisited))
        chosen_node = unvisited_list[action]

        # Add chosen_node to the tour
        from_node = self.tour[-1]
        to_node = chosen_node
        dist = self.distance(from_node, to_node)
        self.current_length += dist
        self.tour.append(chosen_node)
        self.unvisited.remove(chosen_node)

        # If no nodes remain unvisited, the tour is complete.
        # Add distance from the last node back to the first node to close the tour.
        if len(self.unvisited) == 0:
            start_node = self.tour[0]
            end_node = self.tour[-1]
            closing_dist = self.distance(end_node, start_node)
            self.current_length += closing_dist

    def is_terminal(self):
        """
        Terminal when all nodes are visited.
        """
        return len(self.unvisited) == 0

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

    def get_score(self):
        """
        Score could be negative partial length or something similar.
        For partial tours, we might just return negative of current_length.
        """
        return -self.current_length