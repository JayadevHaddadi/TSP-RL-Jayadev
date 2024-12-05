import logging
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from MCTS import MCTS
from TSP.pytorch.NNetWrapper import NNetWrapper

from TSP.TSPGame import TSPGame

import matplotlib.pyplot as plt

import csv

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(
        self,
        game: TSPGame,
        nnet: NNetWrapper,
        args,
        best_tour_length=None,
        folder=None,
    ):
        self.game = game
        self.nnet = nnet
        self.pnet = nnet.__class__(game, args)  # Initialize the previous network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.best_tour_length = best_tour_length
        self.folder = folder

        # Write header to the losses file
        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"]
            )

    def executeEpisode(self):
        trainExamples = []
        tsp_state = initial_tsp_state = self.game.getRandomState()
        episodeStep = 0
        maxSteps = self.args.maxSteps
        final_tour_length = None  # Will be set after the episode ends

        while episodeStep < maxSteps:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(tsp_state, temp=temp)

            # Apply data augmentation by rearranging node and tour order
            symmetries = self.game.getSymmetries(tsp_state, pi)

            for state, pi in symmetries:
                trainExamples.append([state, pi, tsp_state.tour_length])

            action = np.random.choice(len(pi), p=pi)
            tsp_state = self.game.getNextState(tsp_state, action)

        # After the episode ends, compute the final tour length
        final_tour_length = self.game.getTourLength(tsp_state)

        # Assign values to each example based on their potential improvement
        return [
            (
                state_pi_length[0],  # state
                state_pi_length[1],  # pi
                np.clip(
                    2 * ((state_pi_length[2] - final_tour_length) / (state_pi_length[2] + 1e-8)) - 1,
                    -1,
                    1,
                ),
            )
            for state_pi_length in trainExamples
        ]


    def learn(self):
        avg_lengths_new = []
        avg_lengths_old = []
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")
            iterationTrainExamples = []

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # Reset search tree
                iterationTrainExamples.extend(self.executeEpisode())

            # Save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if (
                len(self.trainExamplesHistory)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning(f"Removing the oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            # Backup history to a file
            self.saveTrainExamples(i - 1)

            # Shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # Training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples, iteration=i, losses_file=self.losses_file)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("EVALUATING NEW NETWORK")
            # Evaluate the new network by comparing average tour lengths
            avg_length_old = self.evaluateNetwork(pmcts, name="Old")
            avg_length_new = self.evaluateNetwork(nmcts, name="New")

            # Store the average lengths for plotting
            avg_lengths_old.append(avg_length_old)
            avg_lengths_new.append(avg_length_new)

            log.info(
                f"Average Tour Length - New: {avg_length_new}, Old: {avg_length_old}"
            )

            if avg_length_new < avg_length_old * (1 - self.args.updateThreshold):
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.getCheckpointFile(i)
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )
            else:
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            # After training is complete, print the tour lengths

            # Generate a tour using the new network
            tsp_state = self.game.getRandomState()
            best_tsp_state = tsp_state
            for _ in range(self.args.maxSteps):
                pi = self.mcts.getActionProb(tsp_state, temp=0)
                action = np.argmax(pi)
                next_tsp_state = self.game.getNextState(tsp_state, action)
                if next_tsp_state.tour_length > best_tsp_state.tour_length:
                    best_tsp_state = next_tsp_state
                tsp_state = next_tsp_state

            # Plot the tour
            if self.args.visualize:
                title = f"Iteration {i} - New Network Tour"
                save_path = os.path.join(
                    self.folder,
                    f"{self.game.num_nodes}_nodes_{self.game.node_type}_iteration_{i}_tour.png",
                )
                self.game.plotTour(best_tsp_state, title=title, save_path=save_path)

        print("Average Tour Lengths per Iteration:")
        for iteration, (old_len, new_len) in enumerate(
            zip(avg_lengths_old, avg_lengths_new), 1
        ):
            print(
                f"Iteration {iteration}: Old Avg Length = {old_len}, New Avg Length = {new_len}"
            )

        # Save average tour lengths to a file
        iterations = range(1, self.args.numIters + 1)

        if self.args.visualize:
            plt.figure()
            plt.plot(iterations, avg_lengths_old, label="Old Network")
            plt.plot(iterations, avg_lengths_new, label="New Network")
            plt.xlabel("Iteration")
            plt.ylabel("Average Tour Length")
            plt.title("Average Tour Lengths Over Iterations")
            plt.legend()
            plt.grid(True)
            # Save the plot
            plot_filename = os.path.join(
                self.folder,
                f"{self.game.num_nodes}_nodes_{self.game.node_type}_avg_tour_lengths.png",
            )
            plt.savefig(plot_filename)
            plt.close()

    def evaluateNetwork(self, mcts, name=""):
        total_length = 0
        num_episodes = self.args.numEpsEval
        for _ in tqdm(range(num_episodes), desc="Evaluating " + name + " network"):
            board = self.game.getRandomState()
            for _ in range(self.args.maxSteps):
                canonicalBoard = self.game.getCanonicalForm(board)
                pi = mcts.getActionProb(canonicalBoard, temp=0)
                action = np.argmax(pi)
                board = self.game.getNextState(board, action)
            total_length += self.game.getTourLength(board)
        avg_length = total_length / num_episodes

        # Compare with best known solution if available
        best_tour_length = self.best_tour_length  # Store this in the Coach class
        if best_tour_length is not None:
            log.info(f"Best Known Tour Length: {best_tour_length}")
            log.info(f"Average Tour Length: {avg_length}")
            improvement = ((avg_length - best_tour_length) / best_tour_length) * 100
            log.info(f"Percentage above best known: {improvement:.2f}%")
        else:
            log.info(f"Average Tour Length: {avg_length}")

        return avg_length

    def getCheckpointFile(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.args.load_examples_folder_file[0],
            self.args.load_examples_folder_file[1],
        )
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info("Loading done!")
