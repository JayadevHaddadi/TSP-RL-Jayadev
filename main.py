# main.py
import logging
import os
from datetime import datetime
import multiprocessing as mp
import torch
import numpy as np
import glob

from TSPGame import TSPGame
from NNetWrapper import NNetWrapper
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
        tsp_instance = os.path.join(args.base_folder, args.tsp_instance)
        init_coords, best_tour_length, edge_weight_type = load_tsp_instance(
            tsp_instance
        )
        num_nodes = len(init_coords)

        # Create game with explicit node_type
        game = TSPGame(num_nodes, init_coords, node_type=edge_weight_type, args=args)
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

    # Fetch best known tour length from solutions file if available
    instance_name = args["tsp_instance"] if "tsp_instance" in args else None
    solutions_file = os.path.join(os.path.dirname(instance_name), "solutions")
    solutions_file = os.path.join(args.base_folder, solutions_file)
    logging.info(f"Current working directory: {os.path.abspath(os.getcwd())}")
    logging.info(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    logging.info(f"Solutions file: {solutions_file}")
    logging.info(f"Instance name: {instance_name}")

    best_tour_length = None
    if instance_name:
        solutions = read_solutions(solutions_file)
        instance_key = os.path.splitext(os.path.basename(instance_name))[0]
        # print(f"Solutions: {solutions}")
        best_tour_length = solutions.get(instance_key, None)
        print(f"Best tour length for instance {instance_key}: {best_tour_length}")

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
    # Preset configurations for different scenarios
    preset_configs = {
        "light": {  # Fast training, smaller network
            "configuration name": "light",
            "num_channels": 128,
            "num_layers": 4,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "heads": 4,
            "dropout": 0.1,
            "policy_layers": 1,
            "value_layers": 1,
            "edge_dim": 32,
            "readout_dim": 128,
        },
        "medium": {  # Balanced performance/speed
            "configuration name": "medium",
            "num_channels": 256,
            "num_layers": 8,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "heads": 8,
            "dropout": 0.1,
            "policy_layers": 2,
            "value_layers": 2,
            "edge_dim": 64,
            "readout_dim": 256,
        },
        "heavy": {  # Best performance, slower training
            "configuration name": "heavy",
            "num_channels": 512,
            "num_layers": 12,
            "embedding_dim": 256,
            "hidden_dim": 512,
            "heads": 16,
            "dropout": 0.2,
            "policy_layers": 3,
            "value_layers": 3,
            "edge_dim": 128,
            "readout_dim": 512,
        },
    }
    # Direct assignment of configuration values to base_args
    base_args = dotdict(
        {
            #####################################
            # Basic Configuration
            #####################################
            "configuration name": "standard",
            "explicit_prints": False,
            "base_folder": ".",  # "/home/swaminathanj/jayadev_tsp/alpha_tsp/" or "."
            # TSP Instance Settings
            "tsp_instance": "tsplib/burma14.tsp",  # None,  # Set to path like "tsplib/burma14.tsp" or None for random
            "num_nodes": 5,  # Only used if tsp_instance is None
            #####################################
            # Neural Network Architecture
            #####################################
            # Core Architecture
            "architecture": "gcn",  # Options: "gcn", "pointer", "transformer", "conformer", "graphpointer"
            "num_channels": 128,  # Width of network (64-1024). Larger = more expressive but slower
            "num_layers": 4,  # Number of GCN layers (2-16). Deeper = larger receptive field
            "embedding_dim": 128,  # Initial node embedding size (32-256)
            "hidden_dim": 128,  # Hidden layer width (64-512)
            # Attention & Regularization
            "heads": 8,  # Number of attention heads (4-16). More heads = finer-grained attention
            "dropout": 0.1,  # Dropout rate (0.0-0.5). Higher = more regularization
            "activation": "relu",  # Activation function: 'relu', 'gelu', 'elu'
            # Architecture Components
            "layer_norm": True,  # Whether to use layer normalization
            "skip_connections": True,  # Whether to use residual connections
            "pooling": "mean",  # Graph pooling: 'mean', 'sum', 'max'
            "feature_norm": "batch",  # Feature normalization: 'batch', 'layer', 'none'
            #####################################
            # Training Parameters
            #####################################
            # Optimization
            "learning_rate": 0.01,  # Initial learning rate (1e-4 to 1e-2)
            "pi_lr": 0.00001,  # Initial learning rate for policy network
            "lr_decay": 0.9,  # Learning rate decay factor (0.9-0.99)
            "lr_step_size": 5,
            "weight_decay": 1e-4,  # L2 regularization (1e-5 to 1e-3)
            "grad_clip": 5.0,  # Gradient clipping threshold
            "batch_norm": True,  # Use batch normalization
            "max_gradient_norm": 5.0,  # Maximum gradient norm for clipping
            # Training Loop
            "numIters": 1000,  # Number of training iterations
            "numEps": 4,  # Episodes per iteration
            "epochs": 5,  # Training epochs per iteration
            "batch_size": 64,  # Batch size for training
            #####################################
            # Policy & Value Networks
            #####################################
            # Policy Head
            "policy_layers": 1,  # Number of layers in policy head (1-3)
            "policy_dim": 256,  # Policy head hidden dimension (64-512)
            # Value Head
            "value_layers": 1,  # Number of layers in value head (1-3)
            "value_dim": 256,  # Value head hidden dimension (64-512)
            #####################################
            # Advanced Architecture Options
            #####################################
            # Edge Features
            "use_edge_features": True,  # Whether to use edge features
            "edge_dim": 32,  # Edge feature dimension (16-128)
            # Global Information
            "global_pool": "mean",  # Global pooling method: 'mean', 'sum', 'max'
            "readout_layers": 2,  # Number of layers in readout MLP
            "readout_dim": 128,  # Readout hidden dimension
            # Initialization
            "init_type": "kaiming",  # Weight initialization: 'kaiming', 'xavier', 'orthogonal'
            "init_scale": 1.0,  # Scale factor for initialization
            #####################################
            # MCTS & Evaluation Parameters
            #####################################
            "numMCTSSims": 25,  # Number of MCTS simulations during training
            "numMCTSSimsEval": 25,  # Number of MCTS simulations during evaluation
            "maxlenOfQueue": 200000,  # Maximum length of the queue
            "cpuct": 1.0,  # Exploration constant in MCTS
            "no_improvement_threshold": 3,
            "cpuct_update_factor": 1.05,
            # Training History
            "numItersForTrainExamplesHistory": 20,
            "augmentationFactor": 1,  # Data augmentation factor
            # Evaluation
            "coordinatesToEvaluate": 5,
            "plot_all_eval_sets_interval": 10,
            # System
            "cuda": torch.cuda.is_available(),
            "visualize": True,
            "load_model": False,  # Set True to load a pre-trained model
            "load_folder_file": [
                "for analysis",
                "best burma14 light.pth.tar",  # best burma14 light.pth.tar,best EIL51 6 procent off sol.tar
            ],
            "fixed_start": True,  # Set to False for random starts
            "fixed_start_node": 0,  # Which node to use when fixed_start=True
            "numEpisodesParallel": 8,
            # "numSelfPlayEpisodes": 4,
            "buffer_size": 200000,  # Replay buffer size
            "checkpoint_interval": 5,  # How often to evaluate and save
        }
    )

    # Example of using a preset configuration:
    arch_list = [
        (
            "gat_test",  # Name of the experiment
            {
                "architecture": "gat",
                "embedding_dim": 128,
                "heads": 8,
                "num_layers": 4,  # Example GAT config
                "global_pool": "mean",
                # ... other overrides ...
            },
        ),
        (
            "gpn_test",
            {
                "architecture": "graphpointer",
                # ... gpn config ...
            },
        ),
        (
            "gcn_normal",  # Renamed from "normal" for clarity
            {
                "architecture": "gcn",
                # ... gcn config ...
            },
        ),
        (
            "transformer",  # Renamed from "normal" for clarity
            {
                "architecture": "transformer",
                # ... gcn config ...
            },
        ),
        (
            "pointer",  # Renamed from "normal" for clarity
            {
                "architecture": "pointer",
                # ... gcn config ...
            },
        ),
        (
            "conformer",  # Renamed from "normal" for clarity
            {
                "architecture": "conformer",
                # ... gcn config ...
            },
        ),
        # ... other experiments ...
    ]

    # Generate ONE shared set of evaluation TSPs for consistent comparison
    coords_for_eval = []
    nn_lengths_for_eval = []

    if base_args.tsp_instance:
        # Use the specific TSP instance for evaluation
        tsp_instance = os.path.join(base_args.base_folder, base_args.tsp_instance)
        instance_coords, _, _ = load_tsp_instance(tsp_instance)

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
        #         return
        # else:
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

        run_folder = os.path.join(base_args.base_folder, run_name)
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
