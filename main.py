# main.py
import logging
import os
from datetime import datetime
import multiprocessing as mp
import torch
import numpy as np

from TSP.TSPGame import TSPGame
from TSP.pytorch.NNetWrapper import NNetWrapper
from Coach import Coach
from utils import *

def run_experiment(args, run_folder, coords_for_eval, nn_lengths_for_eval):
    """
    Worker function for parallel execution.

    Each architecture will run in a separate process using
    the same coords_for_eval & nn_lengths_for_eval to ensure
    consistent comparison on evaluation TSPs.
    """
    # Re-init logging for each process (otherwise logs may conflict)
    log_file = os.path.join(run_folder, "log.txt")
    setup_logging(log_file)

    # Convert args to a dotdict if needed
    args = dotdict(args)

    logging.info(f"Starting experiment with architecture: {args.architecture}...")
    logging.info(f"Run folder: {run_folder}")
    logging.info(f"CUDA Available: {args.cuda}")
    
    # Create a TSPGame with random initial coordinates
    # (unless you want them to be consistent for all architectures)
    num_nodes = args.num_nodes
    init_coords = np.random.rand(num_nodes, 2).tolist()
    game = TSPGame(num_nodes, init_coords, args)
    game.node_type = "rand"

    # Suppose you have no best known solution
    best_tour_length = None

    # Initialize the neural network wrapper
    nnet = NNetWrapper(game, args)
    if args.load_model:
        logging.info(f"Loading model from: {args.load_folder_file}")
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    # Create the coach
    coach = Coach(
        game=game,
        nnet=nnet,
        args=args,
        best_tour_length=best_tour_length,
        folder=run_folder,
        coords_for_eval=coords_for_eval,
        nn_lengths_for_eval=nn_lengths_for_eval
    )
    # Run training
    coach.learn()

def main():
    # Base arguments
    base_args = {
        "numIters": 1000,  # short test
        "numEps": 5,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 25,
        "numMCTSSimsEval": 25,
        "coordinatesToEvaluate": 5,
        "plot_all_eval_sets_interval": 10,
        "cpuct": 1,
        "load_model": False,
        "augmentationFactor": 1,
        "numItersForTrainExamplesHistory": 20,

        "architecture": "gcn",
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 5,
        "batch_size": 64,
        "cuda": torch.cuda.is_available(),
        "num_channels": 128,
        "max_gradient_norm": 5.0,
        "visualize": True,
        "read_from_file": False,
        "num_nodes": 20,   # example
    }

    # Different architectures to compare
    # Provide any extra parameters for each arch
    arch_list = [
        ("gcn dropout 0.3",           {"architecture": "gcn",}),
        ("gcn dropout 0.1",           {"architecture": "gcn","dropout": 0.1}),
        ("gcn dropout 0.5",           {"architecture": "gcn","dropout": 0.5}),
        # ("gcn MCTS-50",           {"architecture": "gcn", "numMCTSSimsEval": 50 ,}),
        # ("gcn Augmentation-10",           {"architecture": "gcn",          "augmentationFactor": 10}),
        # ("pointer",       {"architecture": "pointer",      "num_channels": 128, "dropout": 0.3}),
        # ("transformer",   {"architecture": "transformer_deepseek", "num_channels": 256, "dropout": 0.1}),
    ]

    # --------------------------------------------------------------------------
    # Generate ONE shared set of evaluation TSPs so we can compare arch results
    # --------------------------------------------------------------------------
    # e.g. each TSP is 8 random nodes
    common_num_nodes = base_args["num_nodes"]
    coords_for_eval = []
    nn_lengths_for_eval = []
    for _ in range(base_args["coordinatesToEvaluate"]):
        cset = np.random.rand(common_num_nodes, 2).tolist()
        coords_for_eval.append(cset)
        nn_len, _ = compute_nn_tour(cset)
        nn_lengths_for_eval.append(nn_len)
    # Now coords_for_eval & nn_lengths_for_eval are the same for all processes

    processes = []
    for (arch_name, arch_params) in arch_list:
        # Copy base arguments and update with the architecture-specific fields
        these_args = dict(base_args)  # shallow copy
        these_args.update(arch_params)

        # Create a unique run folder
        run_timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        run_name = f"{run_timestamp}_{these_args['num_nodes']}_{arch_name}"
        run_folder = os.path.join("runs", run_name)
        os.makedirs(run_folder, exist_ok=True)

        # Subfolder for checkpoints
        ckpt_folder = os.path.join(run_folder, "checkpoints")
        os.makedirs(ckpt_folder, exist_ok=True)
        these_args["checkpoint"] = ckpt_folder

        # Spawn a process
        p = mp.Process(
            target=run_experiment,
            args=(these_args, run_folder, coords_for_eval, nn_lengths_for_eval)
        )
        processes.append(p)
        p.start()

    # Wait for all processes
    for p in processes:
        p.join()

if __name__ == "__main__":
    # For CUDA + multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
