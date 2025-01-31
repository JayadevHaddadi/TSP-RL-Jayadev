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
            "numMCTSSims": 50,
            "numMCTSSimsEval": 1,
            "coordinatesToEvaluate": 5,
            "plot_all_eval_sets_interval": 10,
            "cpuct": 1,
            "load_model": True,
            "load_folder_file": (
                "./runs/250114-200221_10_rand/checkpoints",
                "best.pth.tar",
            ),
            "augmentationFactor": 20,
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

    # Subfolder for checkpoints
    nn_folder = os.path.join(run_folder, "checkpoints")
    os.makedirs(nn_folder, exist_ok=True)

    # Set up checkpoint path
    args.checkpoint = nn_folder

    log_file = os.path.join(run_folder, "log.txt")
    setup_logging(log_file)

    logging.info(f"Run folder: {run_folder}")
    logging.info("CUDA Available: %s", torch.cuda.is_available())

    # Create initial TSPGame with random coordinates
    num_nodes = args.num_nodes
    init_coords = np.random.rand(num_nodes, 2).tolist()
    game = TSPGame(num_nodes, init_coords, args)
    game.node_type = "rand"

    # Suppose no known best solution
    best_tour_length = None

    # Build coords_for_eval for stable evaluation
    coords_for_eval = []
    nn_lengths_for_eval = []  # <-- We will store the nearest–neighbor lengths here
    for _ in range(args.coordinatesToEvaluate):
        cset = np.random.rand(num_nodes, 2).tolist()
        coords_for_eval.append(cset)
        # Compute NN length for this cset
        nn_len, _ = compute_nn_tour(cset)
        nn_lengths_for_eval.append(nn_len)

    # Initialize your neural network
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

    # Create the Coach with coords_for_eval and the associated NN lengths
    c = Coach(
        game=game,
        nnet=nnet,
        args=args,
        best_tour_length=best_tour_length,
        folder=run_folder,
        coords_for_eval=coords_for_eval,
        nn_lengths_for_eval=nn_lengths_for_eval
    )

    # Run training
    c.learn()

if __name__ == "__main__":
    main()
