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
        self.old_net = nnet.__class__(game, args)  # previous network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.best_tour_length = best_tour_length
        self.folder = folder

        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"]
            )

    def executeEpisode(self):
        trainExamples = []
        tsp_state = self.game.getRandomState()
        episodeStep = 0
        maxSteps = self.args.maxSteps

        while episodeStep < maxSteps:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(tsp_state, temp=temp)

            # Data augmentation symmetries
            symmetries = self.game.getSymmetries(tsp_state, pi)
            for state, pi in symmetries:
                trainExamples.append([state, pi, tsp_state.tour_length])

            action = np.random.choice(len(pi), p=pi)
            tsp_state = self.game.getNextState(tsp_state, action)

        # After the episode ends, compute the final tour length
        final_tour_length = self.game.getTourLength(tsp_state)

        # For each state recorded, compute the value as improvement from that state's tour length
        # relative to the final tour length.
        # value = ((current_tour_length - final_tour_length) / current_tour_length)
        # Clip to [-1, 1]
        new_trainExamples = []
        for state, pi, current_length in trainExamples:
            value = (current_length - final_tour_length) / (current_length + 1e-8)
            value = np.clip(value, -1, 1)
            new_trainExamples.append((state, pi, value))

        return new_trainExamples

    def learn(self):
        avg_lengths_new = []
        avg_lengths_old = []
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")
            iterationTrainExamples = []

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)  # Reset search tree
                iterationTrainExamples.extend(self.executeEpisode())

            # Save iteration examples
            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(f"Removing the oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            # Backup examples
            self.saveTrainExamples(i - 1)

            # Shuffle examples
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # Train new network
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            self.old_net.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

            pmcts = MCTS(self.game, self.old_net, self.args)
            self.nnet.train(trainExamples, iteration=i, losses_file=self.losses_file)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("EVALUATING NEW NETWORK")
            best_state_old = self.evaluateNetwork(pmcts, name="Old")
            best_state_new = self.evaluateNetwork(nmcts, name="New")

            avg_lengths_old.append(best_state_old.tour_length)
            avg_lengths_new.append(best_state_new.tour_length)

            log.info(
                f"Average Tour Length - New: {best_state_new.tour_length}, Old: {best_state_old.tour_length}"
            )

            if best_state_new.tour_length < best_state_old.tour_length * (1 - self.args.updateThreshold):
                best_so_far = best_state_new
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")
            else:
                best_so_far = best_state_old
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

            # Plot the best tour
            if self.args.visualize:
                rounded_len = round(best_so_far.tour_length,4)
                # Add node order to the filename
                tour_str = "_".join(map(str, best_so_far.tour))
                title = f"Iter: {i} - Length: {rounded_len}"
                save_path = os.path.join(
                    self.folder,
                    "graphs",
                    f"Iter_{i:04}_len_{rounded_len:05}_{tour_str}.png",
                )
                self.game.plotTour(best_so_far, title=title, save_path=save_path)

        print("Average Tour Lengths per Iteration:")
        for iteration, (old_len, new_len) in enumerate(zip(avg_lengths_old, avg_lengths_new), 1):
            print(f"Iteration {iteration}: Old Avg Length = {old_len}, New Avg Length = {new_len}")

        if self.args.visualize:
            iterations = range(1, self.args.numIters + 1)
            plt.figure()
            plt.plot(iterations, avg_lengths_old, label="Old Network")
            plt.plot(iterations, avg_lengths_new, label="New Network")
            plt.xlabel("Iteration")
            plt.ylabel("Average Tour Length")
            plt.title("Average Tour Lengths Over Iterations")
            plt.legend()
            plt.grid(True)
            plot_filename = os.path.join(
                self.folder,
                f"{self.game.num_nodes}_nodes_{self.game.node_type}_avg_tour_lengths.png",
            )
            plt.savefig(plot_filename)
            plt.close()

    def evaluateNetwork(self, mcts, name=""):
        tsp_state = self.game.getRandomState()
        best_tsp_state = tsp_state
        for _ in range(self.args.maxSteps):
            pi = mcts.getActionProb(tsp_state, temp=0)
            action = np.argmax(pi)
            next_tsp_state = self.game.getNextState(tsp_state, action)
            if next_tsp_state.tour_length > best_tsp_state.tour_length:
                best_tsp_state = next_tsp_state
            tsp_state = next_tsp_state

        best_current = best_tsp_state.tour_length

        best_tour_length = self.best_tour_length
        if best_tour_length is not None:
            log.info(f"Best Known Tour Length: {best_tour_length}")
            log.info(f"Current best Length for {name}: {best_current}")
            improvement = ((best_current - best_tour_length) / best_tour_length) * 100
            log.info(f"Percentage above best known: {improvement:.2f}%")
        else:
            log.info(f"Current best Length for {name}: {best_current}")

        return best_tsp_state

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

            self.skipFirstSelfPlay = True
