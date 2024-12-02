import logging
import coloredlogs
import torch
import numpy as np

from TSP.TSPGame import TSPGame as Game
from TSP.pytorch.NNetWrapper import NNetWrapper as neural_net_wrapper
from Coach import Coach
from utils import *
import os


log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")

args = dotdict(
    {
        "numIters": 1000, #1000
        "numEps": 10, #100
        "tempThreshold": 15,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 25,  # 25 original value
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./temp", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "maxSteps": 50,  # Maximum steps per episode
        "numEpsEval": 3,  # Number of episodes for evaluation #20
        "updateThreshold": 0.01,  # Minimum improvement threshold (e.g., 1%)

        # New updates from ChatGTP
        "maxDepth": 50,
        "arenaCompare": 40,  # can be removed?

        # For neural Network
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "cuda": torch.cuda.is_available(),
        "num_channels": 128,
        "max_gradient_norm": 5.0,  # Optional: For gradient clipping

        'visualize': True,  # Set to False to disable plotting

        'read_from_file': True,
        'file_name': 'tsplib/burma14.tsp',
        'num_nodes': 6, # For random nodes generation

        'num_processes': 4,  # Adjust based on your CPU cores
    }
)

def main():
    log.info("CUDA Available: %s", torch.cuda.is_available())

    if args.read_from_file:
        node_coords = read_tsplib(args.file_name)
        num_nodes = len(node_coords)

        solutions_file = 'tsplib/solutions'  # Adjust the path as needed
        best_solutions = read_solutions(solutions_file)

        problem_name = os.path.splitext(os.path.basename(args.file_name))[0]

        # Get the best known tour length
        best_tour_length = best_solutions.get(problem_name, None)
        if best_tour_length is None:
            log.info(f"No best known solution found for {problem_name}.")
        else:
            log.info(f"Best known tour length for {problem_name}: {best_tour_length}")

    else:
        # Define node coordinates for TSP
        num_nodes = args.num_nodes  # Adjust the number of nodes as needed
        node_coords = np.random.rand(num_nodes, 2).tolist()  # List of (x, y) tuples

    log.info("Initializing %s...", Game.__name__)
    game = Game(num_nodes, node_coords)  # Initialize TSP game with node coordinates

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

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
