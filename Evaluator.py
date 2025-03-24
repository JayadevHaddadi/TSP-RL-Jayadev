import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from NNetWrapper import NNetWrapper
from MCTS import MCTS
import logging
from TSPGame import TSPGame
from utils import *


class Evaluator:
    def __init__(
        self, game: TSPGame, nnet, args, node_coords, visualize=False, output_folder="evaluation"
    ):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.node_coords = node_coords
        self.visualize = visualize
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        # Initialize MCTS
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def evaluate(self):
        # Initialize state
        state = self.game.getInitState()
        step = 0
        max_steps = self.args.maxSteps

        # Prepare for visualization
        if self.visualize:
            plt.figure()

        while step < max_steps:
            step += 1
            logging.info(f"Step {step}")

            # Get action probabilities
            pi = self.mcts.getActionProb(state, temp=0)

            # Select the best action
            action = np.argmax(pi)
            logging.info(f"Selected action: {action}")

            # Get next state
            next_state = self.game.getNextState(state, action)

            # Optionally visualize
            if self.visualize:
                self.plot_state(next_state, step, action)

            # Update state
            state = next_state

        # Compute final tour length
        final_tour_length = self.game.getTourLength(state)
        logging.info(f"Final tour length: {final_tour_length}")

    def plot_state(self, state, step, action):
        coords = np.array(self.node_coords)
        tour = state.tour + [state.tour[0]]  # Complete the loop

        plt.clf()
        plt.plot(coords[:, 0], coords[:, 1], "o", markersize=10)
        for i in range(len(tour) - 1):
            from_node = tour[i]
            to_node = tour[i + 1]
            plt.plot(
                coords[[from_node, to_node], 0], coords[[from_node, to_node], 1], "r-"
            )

        plt.title(f"Step {step}: Action {action}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(self.output_folder, f"step_{step}.png")
        plt.savefig(plot_filename)

        # Optionally, show the plot (commented out to prevent blocking)
        # plt.show()


args = dotdict(
    {
        "architecture": "gcn",  # Options: "gcn", "pointer", "transformer", "conformer"
        "numIters": 1000,
        "numEps": 10,
        "tempThreshold": 15,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 25,
        "cpuct": 1,
        "checkpoint": "./runs/250321-141833_burma14_light Near SOL/checkpoints",  # Will be updated later
        # runs/6_nodes_random_20241205-104555/checkpoints
        "load_model": False,
        "load_folder_file": ("./temp", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
        "maxSteps": 50,
        "numEpsEval": 2,
        "updateThreshold": 0.01,
        # New updates
        "maxDepth": 50,
        # Neural Network parameters
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "cuda": torch.cuda.is_available(),
        "num_channels": 128,
        "max_gradient_norm": 5.0,
        "visualize": True,
        "read_from_file": False,
        "file_name": "tsplib/burma14.tsp",
        "num_nodes": 20,
    }
)


def evaluate_trained_model():
    # Load node coordinates
    if args.read_from_file:
        node_coords = read_tsplib(args.file_name)
    else:
        node_coords = np.random.rand(args.num_nodes, 2).tolist()

    # Initialize game and network
    game = TSPGame(len(node_coords), node_coords)
    nnet = NNetWrapper(game, args)
    # Load the trained model
    nnet.load_checkpoint(args.checkpoint, filename="best.pth.tar")

    # Create evaluator
    evaluator = Evaluator(
        game, nnet, args, node_coords, visualize=True, output_folder="evaluation_output/eval2"
    )

    # Run evaluation
    evaluator.evaluate()


if __name__ == "__main__":
    evaluate_trained_model()
