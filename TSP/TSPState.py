import numpy as np

class TSPState:
    def __init__(self, num_nodes, node_coordinates):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.tour = [0]
        self.unvisited = np.ones(num_nodes, dtype=int)
        self.unvisited[0] = 0
        self.current_length = 0.0

    def execute_action(self, action):
        """
        Action = node ID (chosen_node).
        unvisited[node] = 1 means node is unvisited.
        """
        chosen_node = action
        if self.unvisited[chosen_node] != 1:
            raise ValueError(f"Chosen action {chosen_node} is not unvisited.")

        # 1) Add the chosen node
        from_node = self.tour[-1]
        to_node = chosen_node
        dist = self.distance(from_node, to_node)
        self.current_length += dist
        self.tour.append(chosen_node)
        self.unvisited[chosen_node] = 0

        # 2) Check how many unvisited remain
        remaining_unvisited = np.sum(self.unvisited)
        if remaining_unvisited == 1:
            # Exactly one node remains, so automatically add it
            last_node = np.where(self.unvisited == 1)[0][0]
            dist_2 = self.distance(self.tour[-1], last_node)
            self.current_length += dist_2
            self.tour.append(last_node)
            self.unvisited[last_node] = 0

            # Now no unvisited remain => close the loop
            start_node = self.tour[0]
            end_node = self.tour[-1]
            closing_dist = self.distance(end_node, start_node)
            self.current_length += closing_dist
        
        elif remaining_unvisited == 0:
            raise Exception("Should never have 0 remaining nodes after doing an action")
            # Already terminal if we just took the last node
            # start_node = self.tour[0]
            # end_node = self.tour[-1]
            # closing_dist = self.distance(end_node, start_node)
            # self.current_length += closing_dist

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
        x1, y1 = self.node_coordinates[i]
        x2, y2 = self.node_coordinates[j]
        return np.hypot(x2 - x1, y2 - y1)