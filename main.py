import logging
import os
from datetime import datetime

import torch
import numpy as np

from TSP.TSPGame import TSPGame
from TSP.pytorch.NNetWrapper import NNetWrapper as neural_net_wrapper
from Coach import Coach
from utils import *


def main():
    args = dotdict(
        {
            "numIters": 1000,
            "numEps": 20,
            "tempThreshold": 15,
            "maxlenOfQueue": 200000,
            "numMCTSSims": 25,
            "cpuct": 1,
            "checkpoint": "./temp/",  # Will be updated later
            "load_model": True,
            "load_folder_file": (
                "./runs/10_nodes_random_20241219-104235/checkpoints",
                "best.pth.tar",
            ),
            "load_examples_folder_file": (
                "./runs/10_nodes_random_20241219-104235/checkpoints",
                "checkpoint",
            ),
            "numItersForTrainExamplesHistory": 20,
            "numEpsEval": 2,
            "updateThreshold": 0.01,
            # Neural Network parameters
            "lr": 0.001,
            "dropout": 0.3,
            "epochs": 5,  # 10
            "batch_size": 64,
            "cuda": torch.cuda.is_available(),
            "num_channels": 128,
            "max_gradient_norm": 5.0,
            # Nodes
            "visualize": True,
            "read_from_file": True,
            "file_name": "./runs/10_nodes_random_20241219-104235/node_coordinates.txt",
            # For Radom
            "num_nodes": 10,
        }
    )

    # Set up run timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Determine node coordinates and node count
    if args.read_from_file:
        node_coords = read_tsplib(args.file_name)
        num_nodes = len(node_coords)
        node_type = os.path.splitext(os.path.basename(args.file_name))[
            0
        ]  # e.g., 'burma14'
    else:
        # Define node coordinates for TSP
        num_nodes = args.num_nodes
        node_coords = np.random.rand(num_nodes, 2).tolist()  # List of (x, y) tuples
        node_type = "random"

    # Construct run folder name
    run_name = f"{num_nodes}_nodes_{node_type}_{run_timestamp}"
    run_folder = os.path.join("runs", run_name)
    os.makedirs(run_folder, exist_ok=True)

    # Create subfolders
    graphs_folder = os.path.join(run_folder, "graphs")
    os.makedirs(graphs_folder, exist_ok=True)

    nn_folder = os.path.join(run_folder, "checkpoints")
    os.makedirs(nn_folder, exist_ok=True)

    # Update args.checkpoint to point to nn_folder
    args.checkpoint = nn_folder

    # Set up logging
    log_file = os.path.join(run_folder, "log.txt")
    setup_logging(log_file)

    # Now proceed with the rest of the main function
    logging.info("CUDA Available: %s", torch.cuda.is_available())
    logging.info(f"Run folder: {run_folder}")

    # Save node coordinates to a file in the run folder
    node_coords_file = os.path.join(run_folder, "node_coordinates.txt")
    save_node_coordinates(node_coords, node_coords_file)

    # Now, if reading from file, get the best known solution
    if args.read_from_file:
        solutions_file = "tsplib/solutions"  # Adjust the path as needed
        best_solutions = read_solutions(solutions_file)

        problem_name = os.path.splitext(os.path.basename(args.file_name))[0]

        # Get the best known tour length
        best_tour_length = best_solutions.get(problem_name, None)
        if best_tour_length is None:
            logging.info(f"No best known solution found for {problem_name}.")
        else:
            logging.info(
                f"Best known tour length for {problem_name}: {best_tour_length}"
            )
    else:
        best_tour_length = None  # Or set to a large value

    logging.info("Initializing %s...", TSPGame.__name__)
    game = TSPGame(
        num_nodes, node_coords, args
    )  # Initialize TSP game with node coordinates
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
    c = Coach(game, nnet, args, best_tour_length=best_tour_length, folder=run_folder)

    if args.load_model:
        logging.info("Loading training examples from file...")
        c.loadTrainExamples()

    logging.info("Starting the learning process HURRAY")
    c.learn()


if __name__ == "__main__":
    main()
