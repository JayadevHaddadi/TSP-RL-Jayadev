import numpy as np
import os
import logging
import sys


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
    import numpy as np
    import logging

    coords = []
    best_tour_length = None
    dimension = 0
    reading_coords = False

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

    return coords, best_tour_length


def log_configuration(args):
    """
    Log all configuration parameters for reference.

    Args:
        args: Dictionary or dotdict containing configuration parameters
    """
    if not isinstance(args, dotdict):
        args = dotdict(args)

    logging.info("=" * 50)
    logging.info("CONFIGURATION PARAMETERS:")
    logging.info("=" * 50)

    # TSP parameters
    logging.info("TSP PARAMETERS:")
    logging.info(f"  TSP Instance: {args.tsp_instance}")
    logging.info(f"  Number of Nodes: {args.num_nodes}")

    # Neural network parameters
    logging.info("NEURAL NETWORK PARAMETERS:")
    logging.info(f"  Architecture: {args.architecture}")
    logging.info(f"  Dropout: {args.dropout}")
    logging.info(f"  Learning Rate: {args.learning_rate}")
    logging.info(f"  Number of Channels: {args.num_channels}")
    logging.info(f"  Max Gradient Norm: {args.max_gradient_norm}")

    # Training parameters
    logging.info("TRAINING PARAMETERS:")
    logging.info(f"  Number of Iterations: {args.numIters}")
    logging.info(f"  Episodes per Iteration: {args.numEps}")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Batch Size: {args.batch_size}")
    logging.info(f"  History Iterations: {args.numItersForTrainExamplesHistory}")
    logging.info(f"  Augmentation Factor: {args.augmentationFactor}")

    # MCTS parameters
    logging.info("MCTS PARAMETERS:")
    logging.info(f"  MCTS Simulations: {args.numMCTSSims}")
    logging.info(f"  MCTS Eval Simulations: {args.numMCTSSimsEval}")
    logging.info(f"  Max Queue Length: {args.maxlenOfQueue}")
    logging.info(f"  CPUCT: {args.cpuct}")

    # Other parameters
    logging.info("OTHER PARAMETERS:")
    logging.info(f"  CUDA: {args.cuda}")
    logging.info(f"  Visualize: {args.visualize}")
    logging.info(f"  Read from File: {args.read_from_file}")
    logging.info(f"  Load Model: {args.load_model}")
    logging.info("=" * 50)
