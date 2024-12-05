import logging
import os
import sys
from datetime import datetime

import torch
import numpy as np

from TSP.TSPGame import TSPGame as Game
from TSP.pytorch.NNetWrapper import NNetWrapper as neural_net_wrapper
from Coach import Coach
from utils import *

def save_node_coordinates(node_coords, filename):
    """
    Saves node coordinates to a file in the TSPLIB format.
    """
    with open(filename, 'w') as f:
        f.write("NAME: Generated\n")
        f.write("TYPE: TSP\n")
        f.write("DIMENSION: {}\n".format(len(node_coords)))
        f.write("NODE_COORD_SECTION\n")
        for idx, (x, y) in enumerate(node_coords, start=1):
            f.write(f"{idx} {x} {y}\n")
        f.write("EOF\n")

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

def main():
    args = dotdict(
        {
            "numIters": 1000,
            "numEps": 10,
            "tempThreshold": 15,
            "maxlenOfQueue": 200000,
            "numMCTSSims": 25,
            "cpuct": 1,
            "checkpoint": "./temp/",  # Will be updated later
            "load_model": False,
            "load_folder_file": ("./temp", "best.pth.tar"),
            "numItersForTrainExamplesHistory": 20,
            "maxSteps": 50,
            "numEpsEval": 2,
            "updateThreshold": 0.01,

            # New updates
            "maxDepth": 50,

            # Neural Network parameters
            "lr": 0.001,
            "dropout": 0.3,
            "epochs": 10,
            "batch_size": 64,
            "cuda": torch.cuda.is_available(),
            "num_channels": 128,
            "max_gradient_norm": 5.0,

            'visualize': True,

            'read_from_file': False,
            'file_name': 'tsplib/burma14.tsp',
            'num_nodes': 6,
        }
    )

    # Set up run timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Determine node coordinates and node count
    if args.read_from_file:
        node_coords = read_tsplib(args.file_name)
        num_nodes = len(node_coords)
        node_type = os.path.splitext(os.path.basename(args.file_name))[0]  # e.g., 'burma14'
    else:
        # Define node coordinates for TSP
        num_nodes = args.num_nodes
        node_coords = np.random.rand(num_nodes, 2).tolist()  # List of (x, y) tuples
        node_type = 'random'

    # Construct run folder name
    run_name = f"{num_nodes}_nodes_{node_type}_{run_timestamp}"
    run_folder = os.path.join('runs', run_name)
    os.makedirs(run_folder, exist_ok=True)

    # Create subfolders
    graphs_folder = os.path.join(run_folder, 'graphs')
    os.makedirs(graphs_folder, exist_ok=True)

    nn_folder = os.path.join(run_folder, 'checkpoints')
    os.makedirs(nn_folder, exist_ok=True)

    # Update args.checkpoint to point to nn_folder
    args.checkpoint = nn_folder

    # Set up logging
    log_file = os.path.join(run_folder, 'log.txt')
    setup_logging(log_file)

    # Now proceed with the rest of the main function
    logging.info("CUDA Available: %s", torch.cuda.is_available())
    logging.info(f"Run folder: {run_folder}")

    # Save node coordinates to a file in the run folder
    node_coords_file = os.path.join(run_folder, 'node_coordinates.txt')
    save_node_coordinates(node_coords, node_coords_file)

    # Now, if reading from file, get the best known solution
    if args.read_from_file:
        solutions_file = 'tsplib/solutions'  # Adjust the path as needed
        best_solutions = read_solutions(solutions_file)

        problem_name = os.path.splitext(os.path.basename(args.file_name))[0]

        # Get the best known tour length
        best_tour_length = best_solutions.get(problem_name, None)
        if best_tour_length is None:
            logging.info(f"No best known solution found for {problem_name}.")
        else:
            logging.info(f"Best known tour length for {problem_name}: {best_tour_length}")
    else:
        best_tour_length = None  # Or set to a large value

    logging.info("Initializing %s...", Game.__name__)
    game = Game(num_nodes, node_coords)  # Initialize TSP game with node coordinates
    game.node_type = node_type  # Add node_type attribute to game
    game.num_nodes = num_nodes

    logging.info("Initializing Neural Network: %s...", neural_net_wrapper.__name__)
    nnet = neural_net_wrapper(game, args)

    if args.load_model:
        logging.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        logging.warning("Not loading a checkpoint! Starting from scratch.")

    logging.info("Initializing the Coach...")
    c = Coach(game, nnet, args, best_tour_length=best_tour_length, graphs_folder=graphs_folder)

    if args.load_model:
        logging.info("Loading training examples from file...")
        c.loadTrainExamples()

    logging.info("Starting the learning process 🎉")
    c.learn()

if __name__ == "__main__":
    main()
