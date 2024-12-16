import logging
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

from MCTS import MCTS
from TSP.pytorch.NNetWrapper import NNetWrapper
from TSP.TSPGame import TSPGame

log = logging.getLogger(__name__)


class Coach:
    """
    Executes self-play + learning. Uses TSPGame and NeuralNet. 
    args specified in main.py.
    """

    def __init__(self, game: TSPGame, nnet: NNetWrapper, args, best_tour_length=None, folder=None):
        self.game = game
        self.nnet = nnet
        self.old_net = nnet.__class__(game, args)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.best_tour_length = best_tour_length
        self.folder = folder
        self.L_baseline = self.compute_baseline_length()
        # self.pi_loss_history = []
        # self.v_loss_history = []
        self.iteration_pi_loss_history = []
        self.iteration_v_loss_history = []
        self.avg_lengths_new = []
        self.avg_lengths_old = []

        # Store baseline in args so TSPGame and others can access it
        self.args.L_baseline = self.L_baseline

        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"])

    def compute_baseline_length(self):
        coords = self.game.node_coordinates
        num_nodes = self.game.num_nodes
        visited = {0}
        current = 0
        length = 0.0
        for _ in range(num_nodes-1):
            best_dist = float('inf')
            best_node = None
            for node in range(num_nodes):
                if node not in visited:
                    x1,y1 = coords[current]
                    x2,y2 = coords[node]
                    d = np.hypot(x2 - x1, y2 - y1)
                    if d < best_dist:
                        best_dist = d
                        best_node = node
            visited.add(best_node)
            length += best_dist
            current = best_node
        return length

    def executeEpisode(self):
        tsp_state = self.game.getInitBoard()

        trajectory = []

        # Self-play until terminal or maxSteps
        while not self.game.isTerminal(tsp_state):
            # we can start off by having higher EXPLORATION in early episodes and less later on
            pi = self.mcts.getActionProb(tsp_state, temp=1) 
            trajectory.append((tsp_state, pi))

            action = np.random.choice(len(pi), p=pi)
            tsp_state = self.game.getNextState(tsp_state, action)

        # Terminal or max steps reached
        final_tour_length = self.game.getTourLength(tsp_state)
        raw_value = (self.L_baseline - final_tour_length) / (self.L_baseline + 1e-8)
        value = np.clip(raw_value, -1, 1)

        new_trainExamples = []
        for (st, pi) in trajectory:
            new_trainExamples.append((st, pi, value))

        return new_trainExamples

    def learn(self):
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")
            iterationTrainExamples = []

            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)
                iterationTrainExamples.extend(self.executeEpisode())

            self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("Removing oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(i - 1)

            # Shuffle examples
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            self.old_net.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

            pmcts = MCTS(self.game, self.old_net, self.args)
            final_pi_loss, final_v_loss = self.nnet.train(trainExamples, iteration=i)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # Record final losses for this iteration
            self.iteration_pi_loss_history.append(final_pi_loss)
            self.iteration_v_loss_history.append(final_v_loss)

            log.info("EVALUATING NEW NETWORK")
            best_state_old = self.evaluateNetwork(pmcts, name="Old")
            best_state_new = self.evaluateNetwork(nmcts, name="New")

            self.avg_lengths_old.append(best_state_old.current_length)
            self.avg_lengths_new.append(best_state_new.current_length)

            log.info(
                f"Average Tour Length - New: {best_state_new.current_length}, Old: {best_state_old.current_length}"
            )

            if best_state_new.current_length < best_state_old.current_length * (1 - self.args.updateThreshold):
                best_so_far = best_state_new
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")
            else:
                best_so_far = best_state_old
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

            # Write current iteration's losses and lengths to CSV
            # iteration, final_pi_loss, final_v_loss, avg_length_new, avg_length_old
            with open(self.losses_file, 'a', newline='') as f:
                writer = csv.writer(f)
                # We'll record iteration-level data
                writer.writerow([i, '', '', final_pi_loss, final_v_loss, self.avg_lengths_new[-1], self.avg_lengths_old[-1] if len(self.avg_lengths_old)==i else ''])

            # Update loss and length plot
            self.plot_loss_and_length_history()

            if self.args.visualize:
                rounded_len = round(best_so_far.current_length,4)
                tour_str = "_".join(map(str, best_so_far.tour))
                title = f"Iter: {i} - Length: {rounded_len}"
                save_path = os.path.join(
                    self.folder,
                    "graphs",
                    f"Iter_{i:04}_len_{rounded_len:05}_{tour_str}.png",
                )
                self.game.plotTour(best_so_far, title=title, save_path=save_path)

        print("Average Tour Lengths per Iteration:")
        for iteration, (old_len, new_len) in enumerate(zip(self.avg_lengths_old, self.avg_lengths_new), 1):
            print(f"Iteration {iteration}: Old Avg Length = {old_len}, New Avg Length = {new_len}")

        if self.args.visualize:
            # Already plotted continuously, final plot also updated.
            pass

    def plot_loss_and_length_history(self):
        """
        Plot both losses and tour lengths over iterations.
        We'll have one iteration = one data point for final losses and final lengths.
        We'll also add a horizontal line for L_baseline.

        Left y-axis: losses (pi and value)
        Right y-axis: tour length (avg_lengths_new) and baseline line
        """
        if len(self.iteration_pi_loss_history) == 0:
            return

        iterations = np.arange(1, len(self.iteration_pi_loss_history)+1)

        plt.figure()
        # Left axis for losses
        ax1 = plt.gca()
        ax1.plot(iterations, self.iteration_pi_loss_history, label="Policy Loss", color='blue')
        ax1.plot(iterations, self.iteration_v_loss_history, label="Value Loss", color='orange')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax1.legend(loc='upper left')

        # Right axis for lengths
        ax2 = ax1.twinx()
        ax2.plot(iterations, self.avg_lengths_new, label="New Network Tour Length", color='green')
        # Horizontal line for baseline
        L_baseline = self.args.L_baseline
        ax2.axhline(y=L_baseline, color='red', linestyle='--', label='Baseline (Nearest Neighbor)')
        ax2.set_ylabel("Tour Length")
        ax2.legend(loc='upper right')

        plt.title("Policy/Value Losses and Tour Length per Iteration")

        loss_plot_path = os.path.join(self.folder, "graphs", "loss_and_length_history.png")
        plt.savefig(loss_plot_path)
        plt.close()

    def evaluateNetwork(self, mcts, name=""):
        tsp_state = self.game.getInitBoard()
        best_tsp_state = tsp_state
        for _ in range(self.game.getActionSize()):
            # Before picking action, check if state is terminal
            # if self.game.isTerminal(tsp_state):
            #     # No more moves, break out
            #     break

            pi = mcts.getActionProb(tsp_state, temp=0)
            action = np.argmax(pi)

            next_tsp_state = self.game.getNextState(tsp_state, action)
            if next_tsp_state.current_length > best_tsp_state.current_length:
                best_tsp_state = next_tsp_state
            tsp_state = next_tsp_state

        # After the loop or terminal detection, we have best_tsp_state
        best_current = best_tsp_state.current_length

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
