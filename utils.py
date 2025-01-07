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
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    def update(self, val, n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
def read_tsplib(filename):
    """
    Reads a TSPLIB file and returns the list of node coordinates.
    """
    node_coords = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    start = False
    for line in lines:
        if line.strip() == 'NODE_COORD_SECTION':
            start = True
            continue
        if start:
            if line.strip() == 'EOF':
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
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                name, length = line.strip().split(':')
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
    with open(filename, 'w') as f:
        f.write("NAME: Generated\n")
        f.write("TYPE: TSP\n")
        f.write("DIMENSION: {}\n".format(len(node_coords)))
        f.write("NN_LENGTH: {:.2f}\n".format(NN_length))  # Format NN length as float
        f.write("NN_TOUR: {}\n".format(" ".join(map(str, NN_tour))))  # Format NN tour as space-separated string
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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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