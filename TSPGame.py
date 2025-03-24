import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import math

sys.path.append("..")
from TSPState import TSPState

import logging

log = logging.getLogger(__name__)


class TSPGame:
    def __init__(self, num_nodes, node_coordinates, node_type="rand", args=None):
        self.num_nodes = num_nodes
        self.node_coordinates = node_coordinates
        self.node_type = node_type
        self.args = args

        # Now this will use the correct distance calculation
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        """Precompute all pairwise distances"""
        n = self.num_nodes
        mat = np.zeros((n, n))

        # Log which distance calculation we're using
        logging.info(f"Computing {self.node_type} distance matrix")

        for i in range(n):
            for j in range(n):
                mat[i][j] = self._calculate_distance(i, j)

        # Log sample distances for verification
        if n >= 2:
            logging.info(f"Sample distance [0][1]: {mat[0][1]}")
            logging.info(f"Sample distance [1][0]: {mat[1][0]}")
        return mat

    def _calculate_distance(self, i, j):
        """Calculate distance between two nodes (used for matrix precomputation)"""
        if self.node_type == "GEO":
            # Use the correct TSPLIB GEO calculation
            return geo_distance(self.node_coordinates[i], self.node_coordinates[j])
        elif self.node_type == "EUC_2D":
            # Euclidean distance for random instances
            x1, y1 = self.node_coordinates[i]
            x2, y2 = self.node_coordinates[j]
            return np.hypot(x2 - x1, y2 - y1)
        else:
            raise ValueError(f"Invalid node type: {self.node_type}")

    def getInitState(self, start_node=None):
        if start_node is None:
            # Default to random start if not specified
            start_node = (
                0
                if self.args.get("fixed_start", True)
                else np.random.choice(self.num_nodes)
            )

        return TSPState(
            num_nodes=self.num_nodes,
            node_coordinates=self.node_coordinates,
            distance_matrix=self.distance_matrix,
            start_node=start_node,
        )

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


# def geo_distance(city1, city2):
#     """
#     Compute the TSPLIB GEO distance between two cities.
#     Each city's coordinate is given as (latitude, longitude) in TSPLIB format.
#     The conversion is:
#        Convert coordinate value X into degrees and minutes:
#          degrees = int(X)
#          minutes = X - degrees
#        Then compute the angle in radians:
#          angle = π*(degrees + 5.0*minutes/3.0) / 180
#     Finally, compute the spherical (great-circle) distance using
#          d = R * arccos( cos(lat1)*cos(lat2)*cos(lon1 - lon2) + sin(lat1)*sin(lat2) )
#     and round each edge as: int(d + 0.5)
#     """
#     # Unpack coordinates for both cities
#     lat1, lon1 = city1
#     lat2, lon2 = city2

#     # Convert latitude and longitude into degrees and fractional minutes
#     lat1_deg = int(lat1)
#     lat1_min = lat1 - lat1_deg
#     lat2_deg = int(lat2)
#     lat2_min = lat2 - lat2_deg

#     lon1_deg = int(lon1)
#     lon1_min = lon1 - lon1_deg
#     lon2_deg = int(lon2)
#     lon2_min = lon2 - lon2_deg

#     # Convert to radians using the TSPLIB GEO transformation
#     lat1_rad = math.radians(lat1_deg + 5.0 * lat1_min / 3.0)
#     lat2_rad = math.radians(lat2_deg + 5.0 * lat2_min / 3.0)
#     lon1_rad = math.radians(lon1_deg + 5.0 * lon1_min / 3.0)
#     lon2_rad = math.radians(lon2_deg + 5.0 * lon2_min / 3.0)

#     # TSPLIB recommends an Earth radius for GEO data of 6378.388 km
#     R = 6378.388

#     # Compute the spherical law of cosines component
#     q = math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(
#         lon1_rad - lon2_rad
#     ) + math.sin(lat1_rad) * math.sin(lat2_rad)
#     # Clamp q to avoid floating-point precision issues
#     q = max(min(q, 1.0), -1.0)

#     # Compute the distance between the two cities
#     d = R * math.acos(q)

#     # Round the edge length exactly as in TSPLIB (round to nearest integer)
#     return int(d + 0.5)

def geo_distance(node1, node2):
    """
    Calculate the geographical distance between two nodes.

    Args:
        node1 (tuple): (latitude, longitude) in degrees.
        node2 (tuple): (latitude, longitude) in degrees.

    Returns:
        int: The geographical distance between the two nodes.
    """
    PI = 3.141592
    RRR = 6378.388  # Radius of the Earth in kilometers

    def deg_to_rad(deg):
        return PI * deg / 180.0

    lat1, lon1 = node1
    lat2, lon2 = node2

    rad_lat1 = deg_to_rad(lat1)
    rad_lon1 = deg_to_rad(lon1)
    rad_lat2 = deg_to_rad(lat2)
    rad_lon2 = deg_to_rad(lon2)

    q1 = math.cos(rad_lon1 - rad_lon2)
    q2 = math.cos(rad_lat1 - rad_lat2)
    q3 = math.cos(rad_lat1 + rad_lat2)

    dij = RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3))
    return int(dij + 0.5)

def compute_tour_length(tour, coordinates):
    """
    Given a tour (as a list of city indices) and a list of city coordinates,
    compute the total tour length using the TSPLIB GEO method.
    Each individual edge is computed and rounded prior to summing.
    """
    total = 0
    n = len(tour)
    for i in range(n):
        city_from = coordinates[tour[i]]
        # Ensure the tour is cyclic
        city_to = coordinates[tour[(i + 1) % n]]
        total += geo_distance(city_from, city_to)
    return total
