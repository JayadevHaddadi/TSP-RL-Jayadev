import numpy as np
import logging
import sys
import copy


###################################
# Utility class AverageMeter
###################################
class AverageMeter(object):
    """From original AlphaZero code: computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def read_tsplib(filename):
    """
    Reads a TSPLIB file and returns the list of node coordinates.
    """
    node_coords = []
    with open(filename, "r") as f:
        lines = f.readlines()

    start = False
    for line in lines:
        if line.strip() == "NODE_COORD_SECTION":
            start = True
            continue
        if start:
            if line.strip() == "EOF":
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                node_coords.append((x, y))

    # Normalize coordinates to [0, 1]
    coords_array = np.array(node_coords)
    min_coords = coords_array.min(axis=0)
    max_coords = coords_array.max(axis=0)
    normal = max_coords - min_coords + 1e-8
    normalized_coords = (coords_array - min_coords) / normal
    return normal, normalized_coords.tolist()


def read_solutions(filename):
    """
    Reads the solutions file and returns a dictionary of best known tour lengths.
    """
    solutions = {}
    with open(filename, "r") as f:
        for line in f:
            if ":" in line:
                name, length = line.strip().split(":")
                name = name.strip()
                try:
                    length = float(length.strip())
                    solutions[name] = length
                except:
                    pass
    return solutions


def write_tsplib(filename, node_coordinates):
    """
    Writes node coordinates to a TSPLIB format file.
    """
    num_nodes = len(node_coordinates)
    with open(filename, "w") as f:
        f.write("NAME: Random_TSP\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(node_coordinates, start=1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")


def save_node_coordinates(node_coords, filename, NN_length, NN_tour):
    """
    Saves node coordinates and NN tour/length to a file in the TSPLIB format.
    """
    with open(filename, "w") as f:
        f.write("NAME: Generated\n")
        f.write("TYPE: TSP\n")
        f.write("DIMENSION: {}\n".format(len(node_coords)))
        f.write("NN_LENGTH: {:.2f}\n".format(NN_length))  # Format NN length as float
        f.write(
            "NN_TOUR: {}\n".format(" ".join(map(str, NN_tour)))
        )  # Format NN tour as space-separated string
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(node_coords, start=1):
            f.write(f"{idx} {x} {y}\n")
        f.write("EOF\n")


def compute_nn_tour(coords):
    num_nodes = len(coords)
    visited = {0}
    current = 0
    tour = [0]  # Start with node 0 as the initial tour
    tour_length = 0.0  # Initialize tour length

    for _ in range(num_nodes - 1):
        best_dist = float("inf")
        best_node = None

        for node in range(num_nodes):
            if node not in visited:
                x1, y1 = coords[current]
                x2, y2 = coords[node]
                d = np.hypot(x2 - x1, y2 - y1)

                if d < best_dist:
                    best_dist = d
                    best_node = node

        visited.add(best_node)
        tour.append(best_node)  # Add the best node to the tour
        tour_length += best_dist  # Add the distance to the tour length
        current = best_node

    # Add the last edge from the last visited node back to the starting node (node 0)
    x1, y1 = coords[current]
    x2, y2 = coords[0]  # Starting node
    last_edge = np.hypot(x2 - x1, y2 - y1)

    tour_length += last_edge  # Add the last edge length to the tour length
    tour.append(0)  # Add the starting node to close the tour

    return tour_length, tour


def setup_logging(log_file_path):
    """
    Sets up logging to write both to console and a log file.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    while logger.handlers:
        logger.handlers.pop()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def load_tsp_instance(filepath):
    """
    Load a TSP instance from a TSPLIB file
    Returns coordinates and optionally the best known solution length
    """
    import logging

    coords = []
    best_tour_length = None
    dimension = 0
    reading_coords = False
    EDGE_WEIGHT_TYPE = "GEO"

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Process header information
            if line.startswith("DIMENSION"):
                dimension = int(line.split()[-1])
            elif line.startswith("BEST_KNOWN"):
                best_tour_length = float(line.split()[-1])
            # Start reading coordinates
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                coords = []  # Initialize empty list for coordinates
            # End of file
            elif line == "EOF" or line.startswith("DISPLAY_DATA_SECTION"):
                break
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                EDGE_WEIGHT_TYPE = line.split()[-1]
            # Read coordinates if we're in the coordinates section
            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:  # Node index, x, y
                    x, y = float(parts[1]), float(parts[2])
                    coords.append([x, y])

    # Debug information
    logging.info(f"Read {len(coords)} coordinates from file with dimension {dimension}")

    # Verify we got the expected number of nodes
    if dimension > 0 and len(coords) != dimension:
        logging.warning(f"Expected {dimension} nodes but got {len(coords)}")

        # Print the first few coordinates for debugging
        if coords:
            logging.info(f"First coordinate: {coords[0]}")
            if len(coords) > 1:
                logging.info(f"Second coordinate: {coords[1]}")

    logging.info(
        f"Loaded TSP instance with {len(coords)} nodes (using raw coordinates)"
    )
    if best_tour_length:
        logging.info(f"Best known solution length: {best_tour_length}")

    return coords, best_tour_length, EDGE_WEIGHT_TYPE


def log_configuration(config):
    """Dynamically logs all configuration key-value pairs."""
    for key, value in config.items():
        logging.info(f"{key}: {value}")


def complete_tour_with_nearest_neighbor(tsp_state):
    """
    Given a TSPState, simulates completing a full tour using the greedy nearest neighbor heuristic.

    The function:
    - Makes a deep copy of tsp_state so the original is unmodified.
    - Starting from the current state (using the last visited node in tsp_state.tour),
      repeatedly selects the nearest unvisited node.
    - When no unvisited nodes remain, it adds the distance from the last visited node
      back to the starting node.

    Returns:
        The full tour length as a float.
    """
    # Make a deep copy of tsp_state to avoid modifying the actual state.
    state_copy = copy.deepcopy(tsp_state)

    # If the tour is already complete, simply return the current tour length.
    if state_copy.is_terminal():
        return state_copy.get_tour_length()

    # Initialize the total cost with the current length.
    # total_cost = state_copy.current_length
    additional_cost = 0

    # Determine the current node and the start node.
    if len(state_copy.tour) > 0:
        current_node = state_copy.tour[-1]
        start_node = state_copy.tour[0]
    else:
        # If tour is empty, arbitrarily choose the first node (should not happen normally).
        current_node = 0
        start_node = 0
        state_copy.tour.append(0)
        state_copy.unvisited[0] = 0

    # While there are still unvisited nodes, select the nearest neighbor.
    while np.sum(state_copy.unvisited) > 0:
        unvisited_indices = np.where(state_copy.unvisited == 1)[0]
        if len(unvisited_indices) == 0:
            break

        best_distance = float("inf")
        best_neighbor = None

        # Evaluate all unvisited nodes to find the minimum distance.
        for node in unvisited_indices:
            d = state_copy.distance(current_node, node)
            if d < best_distance:
                best_distance = d
                best_neighbor = node

        # Add the best distance to the total cost.
        additional_cost += best_distance

        # Update the state copy: append the chosen neighbor and mark it as visited.
        state_copy.tour.append(int(best_neighbor))
        state_copy.unvisited[int(best_neighbor)] = 0
        current_node = best_neighbor

    # Finally, add the closing leg from the last node back to the start.
    closing_cost = state_copy.distance(current_node, start_node)
    additional_cost += closing_cost

    return additional_cost


def calculate_tour_length(tour, distance_matrix):
    """
    Given a tour (list of node indices) and a distance_matrix,
    returns the total tour length (including closing the loop).
    """
    total = 0.0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i]][tour[i + 1]]
    total += distance_matrix[tour[-1]][tour[0]]
    return total


def two_opt(tour, distance_matrix):
    """
    A simple 2-opt improvement algorithm.
    Given an initial tour and the distance_matrix,
    iteratively improves the tour by exchanging two edges until no improvement is found.
    Returns the improved tour and its total cost.
    """
    best = tour
    best_distance = calculate_tour_length(tour, distance_matrix)
    improved = True

    while improved:
        improved = False
        # Try swapping every pair of edges (excluding the starting node fixed)
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1:  # Consecutive nodes; no change.
                    continue
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_distance = calculate_tour_length(new_tour, distance_matrix)
                if new_distance < best_distance:
                    best = new_tour
                    best_distance = new_distance
                    improved = True
        tour = best
    return best, best_distance


def complete_tour_with_lin_kernighan(tsp_state):
    """
    Given a TSPState, completes the tour using an initial nearest-neighbor heuristic,
    then improves that solution using a 2-opt local search (as a proxy for Lin–Kernighan).

    The function makes a deep copy of the tsp_state so that the original is unmodified.
    It returns the additional cost (i.e. the full tour length minus the current state's length).
    """
    # Deep copy so we don't modify the original state.
    state_copy = copy.deepcopy(tsp_state)

    # If the tour is already complete, no additional cost is needed.
    if state_copy.is_terminal():
        return 0.0

    additional_cost = 0.0

    # Determine the current node and the starting node.
    if len(state_copy.tour) > 0:
        current_node = state_copy.tour[-1]
        start_node = state_copy.tour[0]
    else:
        # This should not happen normally.
        current_node = 0
        start_node = 0
        state_copy.tour.append(0)
        state_copy.unvisited[0] = 0

    # Complete the tour using the nearest neighbor heuristic.
    while np.sum(state_copy.unvisited) > 0:
        unvisited_indices = np.where(state_copy.unvisited == 1)[0]
        if len(unvisited_indices) == 0:
            break
        best_distance = float("inf")
        best_neighbor = None
        for node in unvisited_indices:
            d = state_copy.distance(current_node, node)
            if d < best_distance:
                best_distance = d
                best_neighbor = node
        additional_cost += best_distance
        state_copy.tour.append(int(best_neighbor))
        state_copy.unvisited[int(best_neighbor)] = 0
        current_node = best_neighbor

    closing_cost = state_copy.distance(current_node, start_node)
    additional_cost += closing_cost

    # Calculate the complete tour length from the current state's perspective.
    initial_tour = state_copy.tour.copy()
    initial_full_cost = state_copy.current_length + additional_cost

    # Now, improve the complete tour using 2-opt.
    improved_tour, improved_full_cost = two_opt(
        initial_tour, state_copy.distance_matrix
    )

    # Calculate the additional cost needed to complete the tour,
    # relative to the current state's length.
    additional_cost_improved = improved_full_cost - state_copy.current_length
    return additional_cost_improved
