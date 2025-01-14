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
            "numEps": 10,
            "maxlenOfQueue": 200000,
            "numMCTSSims": 25,
            "numMCTSSimsEval": 50,
            "coordinatesToEvaluate": 5,
            "plot_all_eval_sets_interval": 10,
            "cpuct": 1,
            "load_model": False,
            "numItersForTrainExamplesHistory": 20,
            # Neural Network parameters
            "lr": 0.001,
            "dropout": 0.3,
            "epochs": 5,
            "batch_size": 64,
            "cuda": torch.cuda.is_available(),
            "num_channels": 128,
            "max_gradient_norm": 5.0,
            # Node options
            "visualize": True,
            "read_from_file": False,
            "num_nodes": 10,
            # Possibly more arguments
        }
    )

    run_timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    run_name = f"{run_timestamp}_{args.num_nodes}_rand"
    run_folder = os.path.join("runs", run_name)
    os.makedirs(run_folder, exist_ok=True)

    # Create subfolders
    eval_folder = os.path.join(run_folder, "evaluation")
    os.makedirs(eval_folder, exist_ok=True)

    nn_folder = os.path.join(run_folder, "checkpoints")
    os.makedirs(nn_folder, exist_ok=True)

    # *** IMPORTANT: Assign to args.checkpoint BEFORE using Coach or saveTrainExamples() ***
    args.checkpoint = nn_folder

    log_file = os.path.join(run_folder, "log.txt")
    setup_logging(log_file)

    logging.info(f"Run folder: {run_folder}")
    logging.info("CUDA Available: %s", torch.cuda.is_available())

    # Create initial TSPGame
    num_nodes = args.num_nodes
    init_coords = np.random.rand(num_nodes, 2).tolist()
    game = TSPGame(num_nodes, init_coords, args)
    game.node_type = "rand"

    best_tour_length = None

    # Suppose you build coords_for_eval for stable evaluation:
    coords_for_eval = []
    # ... create or store random coords for evaluation
    # e.g.:
    for _ in range(args.coordinatesToEvaluate):
        cset = np.random.rand(num_nodes, 2).tolist()
        coords_for_eval.append(cset)

    # Initialize your neural network
    nnet = neural_net_wrapper(game, args)

    # Create the Coach with all the needed arguments
    c = Coach(
        game=game,
        nnet=nnet,
        args=args,
        best_tour_length=best_tour_length,
        folder=run_folder,
        coords_for_eval=coords_for_eval,
    )

    # Run training
    c.learn()


if __name__ == "__main__":
    main()
