import logging
import coloredlogs
import torch
import numpy as np
from datetime import datetime
import os

from TSP.TSPGame import TSPGame as Game
from TSP.pytorch.NNetWrapper import NNetWrapper as neural_net_wrapper
from Coach import Coach
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")

args = dotdict(
    {
        "numIters": 1000,
        "numEps": 2, #100
        "tempThreshold": 15,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 25,
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./temp", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "maxSteps": 50,
        "numEpsEval": 2,
        "updateThreshold": 0.01,
        "maxDepth": 50,
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 2, #10
        "batch_size": 64,
        "cuda": torch.cuda.is_available(),
        "num_channels": 128,
        "max_gradient_norm": 5.0,
        "visualize": True,
        "read_from_file": False,
        "file_name": "tsplib/burma14.tsp",
        "num_nodes": 6,
    }
)


def setup_run_directory():
    """
    Create a unique directory for the current run inside 'runs' folder.
    Includes subfolders for graphs, neural nets, and logs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.num_nodes}_nodes_"
    run_name += (
        "random_"
        if not args.read_from_file
        else f"from_{args.file_name.split('/')[-1]}_"
    )
    run_name += timestamp

    base_path = os.path.join("runs", run_name)
    os.makedirs(base_path, exist_ok=True)

    subfolders = ["graphs", "neural_net"]
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_path, subfolder), exist_ok=True)

    log_file = os.path.join(base_path, "output.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    log.addHandler(file_handler)

    return base_path


def save_initial_data(base_path, node_coords, best_tour_length=None):
    """
    Save initial data like node coordinates and best known solution to a file.
    """
    coords_file = os.path.join(base_path, "node_coordinates.tsp")
    with open(coords_file, "w") as f:
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(node_coords, start=1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")

    if best_tour_length is not None:
        with open(os.path.join(base_path, "best_tour_length.txt"), "w") as f:
            f.write(f"Best Known Tour Length: {best_tour_length}\n")


def main():
    log.info("CUDA Available: %s", torch.cuda.is_available())

    # Setup directory structure for the current run
    base_path = setup_run_directory()

    if args.read_from_file:
        node_coords = read_tsplib(args.file_name)
        num_nodes = len(node_coords)

        solutions_file = "tsplib/solutions"
        best_solutions = read_solutions(solutions_file)

        problem_name = os.path.splitext(os.path.basename(args.file_name))[0]
        best_tour_length = best_solutions.get(problem_name, None)
        if best_tour_length is None:
            log.info(f"No best known solution found for {problem_name}.")
        else:
            log.info(f"Best known tour length for {problem_name}: {best_tour_length}")

    else:
        num_nodes = args.num_nodes
        node_coords = np.random.rand(num_nodes, 2).tolist()
        best_tour_length = None

    log.info("Initializing %s...", Game.__name__)
    game = Game(num_nodes, node_coords)

    log.info("Initializing Neural Network: %s...", neural_net_wrapper.__name__)
    nnet = neural_net_wrapper(game, args)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint! Starting from scratch.")

    log.info("Initializing the Coach...")
    c = Coach(game, nnet, args, best_tour_length=best_tour_length)

    if args.load_model:
        log.info("Loading training examples from file...")
        c.loadTrainExamples()

    # Save initial data
    save_initial_data(base_path, node_coords, best_tour_length)

    # Set paths for graphs and neural net outputs
    c.graphs_folder = os.path.join(base_path, "graphs")
    c.nn_folder = os.path.join(base_path, "neural_net")

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
