import logging
import coloredlogs
import torch
import numpy as np

from Coach import Coach
from TSP.TSPGame import TSPGame as Game
from TSP.pytorch.NNetWrapper import NNetWrapper as neural_net_wrapper
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")

args = dotdict(
    {
        "numIters": 1000,
        "numEps": 100,
        "tempThreshold": 15,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 25, #25 original value
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./temp", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,

        'maxSteps': 50,  # Maximum steps per episode
        'numEpsEval': 20,  # Number of episodes for evaluation
        'updateThreshold': 0.01,  # Minimum improvement threshold (e.g., 1%)
        
        'maxDepth ': 50,
        "updateThreshold": 0.6, # can be removed?
        "arenaCompare": 40, # can be removed?
    }
)


def main():
    log.info("CUDA Available: %s", torch.cuda.is_available())

    # Define node coordinates for TSP
    num_nodes = 10  # Adjust the number of nodes as needed
    node_coords = np.random.rand(num_nodes, 2).tolist()  # List of (x, y) tuples

    log.info("Initializing %s...", Game.__name__)
    g = Game(num_nodes, node_coords)  # Initialize TSP game with node coordinates

    log.info("Initializing Neural Network: %s...", neural_net_wrapper.__name__)
    nnet = neural_net_wrapper(g)

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
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading training examples from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
