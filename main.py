# main.py
import logging
import os
from datetime import datetime
import multiprocessing as mp
import torch
import numpy as np
import glob

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

    # Log all configuration parameters for future reference
    log_configuration(args)

    logging.info(f"Starting experiment with architecture: {args.architecture}...")
    logging.info(f"Run folder: {run_folder}")
    logging.info(f"CUDA Available: {args.cuda}")

    # Create a TSPGame with the appropriate coordinates
    num_nodes = args.num_nodes
    best_tour_length = None

    if args.tsp_instance:
        # Load specific TSP instance
        logging.info(f"Loading TSP instance from: {args.tsp_instance}")
        init_coords, best_tour_length = load_tsp_instance(args.tsp_instance)
        num_nodes = len(init_coords)

        # Create game with explicit node_type
        game = TSPGame(num_nodes, init_coords, node_type="tsplib", args=args)
    else:
        # Use random coordinates
        init_coords = np.random.rand(num_nodes, 2).tolist()
        game = TSPGame(num_nodes, init_coords, args)
        game.node_type = "rand"

    # Validate evaluation coordinates
    for i, coords in enumerate(coords_for_eval):
        if not all(isinstance(coord, list) and len(coord) == 2 for coord in coords):
            logging.error(f"Invalid evaluation coordinates format in set {i}")
            return
        if len(coords) != num_nodes:
            logging.warning(
                f"Evaluation set {i} has {len(coords)} nodes, but game expects {num_nodes}"
            )

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
        nn_lengths_for_eval=nn_lengths_for_eval,
    )
    # Run training
    coach.learn()


def list_tsp_instances():
    """List available TSP instances in the tsplib folder"""
    tsp_files = glob.glob("tsplib/*.tsp")
    if not tsp_files:
        print("No TSP instances found in tsplib/ folder")
        return

    print("Available TSP instances:")
    for i, file_path in enumerate(sorted(tsp_files), 1):
        print(f"  {i}. {os.path.basename(file_path)}")


def main():
    """
    Main function with direct parameter configuration.
    Edit the parameters below to configure your experiment.
    """
    # Direct assignment of configuration values to base_args
    base_args = dotdict(
        {
            # TSP instance parameters
            "tsp_instance": "tsplib/burma14.tsp",  # Set to None for random TSP
            "num_nodes": 20,  # Only used if tsp_instance is None
            # Neural network parameters
            "architecture": "gcn",  # Choose: "gcn", "pointer", or "transformer"
            "dropout": 0.5,
            "lr": 0.001,
            "num_channels": 512,
            "max_gradient_norm": 5.0,
            # Training parameters
            "numIters": 1000,
            "numEps": 40,
            "epochs": 5,
            "batch_size": 64,
            "numItersForTrainExamplesHistory": 20,
            "augmentationFactor": 1,
            # MCTS parameters
            "numMCTSSims": 100,
            "numMCTSSimsEval": 100,
            "maxlenOfQueue": 200000,
            "cpuct": 1.0,
            # Evaluation parameters
            "coordinatesToEvaluate": 5,
            "plot_all_eval_sets_interval": 10,
            # Other parameters
            "cuda": torch.cuda.is_available(),
            "visualize": True,
            "read_from_file": False,
            "load_model": False,  # Set this to True if you want to load a model
        }
    )

    # Different architectures to compare
    arch_list = [
        (
            "burma14 normal",  # Name of the experiment
            {
                # Override any parameters from base_args here
                "numMCTSSims": 25,
                "numMCTSSimsEval": 25,
                "numEps": 5,
                "tsp_instance": "tsplib/burma14.tsp",
            },
        ),
        # Add more configurations here if needed
        # (
        #     "eil51 large",
        #     {
        #         "tsp_instance": "tsplib/eil51.tsp",
        #         "num_channels": 512,
        #     },
        # ),
    ]

    # Generate ONE shared set of evaluation TSPs for consistent comparison
    coords_for_eval = []
    nn_lengths_for_eval = []

    if base_args.tsp_instance:
        # Use the specific TSP instance for evaluation
        instance_coords, _ = load_tsp_instance(base_args.tsp_instance)

        # Ensure instance_coords is properly formatted
        if instance_coords and all(
            isinstance(coord, list) and len(coord) == 2 for coord in instance_coords
        ):
            logging.info(
                f"Evaluation using TSP instance with {len(instance_coords)} nodes"
            )
            coords_for_eval = [instance_coords]  # Single set with instance coordinates
            nn_len, _ = compute_nn_tour(instance_coords)
            nn_lengths_for_eval = [nn_len]
        else:
            logging.error(
                f"Invalid coordinate format in TSP instance: {base_args.tsp_instance}"
            )
            return
    else:
        # Generate random evaluation sets
        common_num_nodes = base_args.num_nodes
        for _ in range(base_args.coordinatesToEvaluate):
            cset = np.random.rand(common_num_nodes, 2).tolist()
            coords_for_eval.append(cset)
            nn_len, _ = compute_nn_tour(cset)
            nn_lengths_for_eval.append(nn_len)

    # Run experiments
    processes = []
    for arch_name, arch_params in arch_list:
        # Copy base arguments and update with the architecture-specific fields
        these_args = dict(base_args)  # shallow copy
        these_args.update(arch_params)

        # Create a unique run folder
        run_timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

        # Include TSP instance name in folder if specified
        if these_args["tsp_instance"]:
            tsp_name = os.path.basename(these_args["tsp_instance"]).replace(".tsp", "")
            run_name = f"{run_timestamp}_{tsp_name}_{arch_name}"
        else:
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
            args=(these_args, run_folder, coords_for_eval, nn_lengths_for_eval),
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
