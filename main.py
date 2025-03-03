# main.py
import logging
import os
from datetime import datetime
import multiprocessing as mp
import torch
import numpy as np
import argparse
import glob

from TSP.TSPGame import TSPGame
from TSP.pytorch.NNetWrapper import NNetWrapper
from Coach import Coach
from utils import *

def load_tsp_instance(filepath):
    """
    Load a TSP instance from a TSPLIB file
    Returns coordinates and optionally the best known solution length
    """
    coords = []
    best_tour_length = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Parse header information
        i = 0
        dimension = 0
        reading_coords = False
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("DIMENSION"):
                dimension = int(line.split()[-1])
            elif line.startswith("BEST_KNOWN"):
                best_tour_length = float(line.split()[-1])
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                i += 1  # Skip to the next line to start reading coordinates
            elif reading_coords:
                if line == "EOF" or line.startswith("DISPLAY_DATA_SECTION"):
                    break
                    
                parts = line.split()
                if len(parts) >= 3:  # Node index, x, y
                    x, y = float(parts[1]), float(parts[2])
                    coords.append([x, y])
            
            i += 1
    
    # Normalize coordinates to [0, 1] range if needed
    if coords:
        coords = np.array(coords)
        min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
        max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
        
        # Apply normalization
        coords[:, 0] = (coords[:, 0] - min_x) / (max_x - min_x) if max_x > min_x else 0.5
        coords[:, 1] = (coords[:, 1] - min_y) / (max_y - min_y) if max_y > min_y else 0.5
        
        coords = coords.tolist()
    
    logging.info(f"Loaded TSP instance with {len(coords)} nodes")
    if best_tour_length:
        logging.info(f"Best known solution length: {best_tour_length}")
    
    return coords, best_tour_length

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
    
    # Create a TSPGame with the appropriate coordinates
    num_nodes = args.num_nodes
    best_tour_length = None
    
    if args.tsp_instance:
        # Load specific TSP instance
        logging.info(f"Loading TSP instance from: {args.tsp_instance}")
        init_coords, best_tour_length = load_tsp_instance(args.tsp_instance)
        args.num_nodes = len(init_coords)
        game = TSPGame(len(init_coords), init_coords, args)
        game.node_type = "tsplib"
    else:
        # Use random coordinates
        init_coords = np.random.rand(num_nodes, 2).tolist()
        game = TSPGame(num_nodes, init_coords, args)
        game.node_type = "rand"

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

def list_tsp_instances():
    """List available TSP instances in the tsplib folder"""
    tsp_files = glob.glob("tsplib/*.tsp")
    if not tsp_files:
        print("No TSP instances found in tsplib/ folder")
        return
    
    print("Available TSP instances:")
    for i, file_path in enumerate(sorted(tsp_files), 1):
        print(f"  {i}. {os.path.basename(file_path)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='TSP Solver with Neural Networks')
    
    # TSP instance options
    parser.add_argument('--tsp-instance', type=str, default="tsplib/eil51.tsp" ,help='Path to a specific TSP instance file in tsplib folder')
    parser.add_argument('--list-instances', action='store_true', help='List available TSP instances in tsplib folder')
    
    # Base arguments - same as your existing defaults
    parser.add_argument('--num-nodes', type=int, default=20, help='Number of nodes for random TSP (ignored if using tsplib instance)')
    parser.add_argument('--num-iters', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--num-eps', type=int, default=5, help='Number of episodes per iteration')
    parser.add_argument('--max-queue-len', type=int, default=200000, help='Maximum length of queue')
    parser.add_argument('--num-mcts-sims', type=int, default=25, help='Number of MCTS simulations')
    parser.add_argument('--num-mcts-sims-eval', type=int, default=25, help='Number of MCTS simulations for evaluation')
    parser.add_argument('--coords-to-evaluate', type=int, default=5, help='Number of coordinate sets to evaluate on')
    parser.add_argument('--plot-interval', type=int, default=10, help='Interval for plotting all evaluation sets')
    parser.add_argument('--cpuct', type=float, default=1.0, help='CPUCT parameter')
    parser.add_argument('--load-model', action='store_true', help='Load an existing model')
    parser.add_argument('--augmentation-factor', type=int, default=1, help='Data augmentation factor')
    parser.add_argument('--history-iters', type=int, default=20, help='Number of iterations for train examples history')
    
    # Neural network arguments
    parser.add_argument('--architecture', type=str, default='gcn', choices=['gcn', 'pointer', 'transformer'], help='Neural network architecture')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-channels', type=int, default=128, help='Number of channels')
    parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Max gradient norm')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    parser.add_argument('--read-from-file', action='store_true', help='Read from file')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle list-instances request
    if args.list_instances:
        list_tsp_instances()
        return
    
    # Convert arguments to dictionary format for compatibility
    base_args = {
        "numIters": args.num_iters,
        "numEps": args.num_eps,
        "maxlenOfQueue": args.max_queue_len,
        "numMCTSSims": args.num_mcts_sims,
        "numMCTSSimsEval": args.num_mcts_sims_eval,
        "coordinatesToEvaluate": args.coords_to_evaluate,
        "plot_all_eval_sets_interval": args.plot_interval,
        "cpuct": args.cpuct,
        "load_model": args.load_model,
        "augmentationFactor": args.augmentation_factor,
        "numItersForTrainExamplesHistory": args.history_iters,

        "architecture": args.architecture,
        "lr": args.lr,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cuda": torch.cuda.is_available() and not args.no_cuda,
        "num_channels": args.num_channels,
        "max_gradient_norm": args.max_gradient_norm,
        "visualize": not args.no_visualize,
        "read_from_file": args.read_from_file,
        "num_nodes": args.num_nodes,
        "tsp_instance": args.tsp_instance,
    }

    # Different architectures to compare
    # Provide any extra parameters for each arch
    arch_list = [
        ("gcn dropout 0.3",           {"architecture": "gcn",}),
        # ("gcn dropout 0.1",           {"architecture": "gcn","dropout": 0.1}),
        # ("gcn dropout 0.5",           {"architecture": "gcn","dropout": 0.5}),
        # ("gcn MCTS-50",           {"architecture": "gcn", "numMCTSSimsEval": 50 ,}),
        # ("gcn Augmentation-10",           {"architecture": "gcn",          "augmentationFactor": 10}),
        # ("pointer",       {"architecture": "pointer",      "num_channels": 128, "dropout": 0.3}),
        # ("transformer",   {"architecture": "transformer_deepseek", "num_channels": 256, "dropout": 0.1}),
    ]

    # --------------------------------------------------------------------------
    # Generate ONE shared set of evaluation TSPs so we can compare arch results
    # --------------------------------------------------------------------------
    # If we're using a specific instance and have multiple architectures,
    # we'll use that instance as the first evaluation set
    coords_for_eval = []
    nn_lengths_for_eval = []
    
    # If using a specific TSP instance, load it for evaluation
       # Generate ONE shared set of evaluation TSPs so we can compare arch results
    coords_for_eval = []
    nn_lengths_for_eval = []
    
    if args.tsp_instance:
        # Use only the specific TSP instance for evaluation
        instance_coords, _ = load_tsp_instance(args.tsp_instance)
        coords_for_eval = [instance_coords]  # Single set with instance coordinates
        nn_len, _ = compute_nn_tour(instance_coords)
        nn_lengths_for_eval = [nn_len]
    else:
        # Generate random evaluation sets when no specific instance is provided
        common_num_nodes = base_args["num_nodes"]
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
        
        # Include TSP instance name in folder if specified
        if args.tsp_instance:
            tsp_name = os.path.basename(args.tsp_instance).replace('.tsp', '')
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