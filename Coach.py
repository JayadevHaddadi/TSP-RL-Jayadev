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

    def __init__(
        self, game: TSPGame, nnet: NNetWrapper, args, best_tour_length=None, folder=None
    ):
        self.game = game
        self.nnet = nnet
        self.old_net = nnet.__class__(game, args)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.best_tour_length = best_tour_length
        self.folder = folder

        # Compute nearest neighbor length as a reference
        self.nn_length = self.compute_nn_length()
        # Set a baseline as simple function of num_nodes
        self.baseline = args.num_nodes / 2

        self.iteration_pi_loss_history = []
        self.iteration_v_loss_history = []
        self.avg_lengths_new = []
        self.avg_lengths_old = []

        # Store baseline in args so TSPGame and others can access it
        self.args.L_baseline = self.baseline

        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"]
            )

    def compute_nn_length(self):
        coords = self.game.node_coordinates
        num_nodes = self.game.num_nodes
        visited = {0}
        current = 0
        length = 0.0
        for _ in range(num_nodes - 1):
            best_dist = float("inf")
            best_node = None
            for node in range(num_nodes):
                if node not in visited:
                    x1, y1 = coords[current]
                    x2, y2 = coords[node]
                    d = np.hypot(x2 - x1, y2 - y1)
                    if d < best_dist:
                        best_dist = d
                        best_node = node
            visited.add(best_node)
            length += best_dist
            current = best_node
        return length

    def executeEpisode(self):
        tsp_state = self.game.getInitState()

        trajectory = []

        # Self-play until terminal or maxSteps (maxSteps not currently used)
        while not self.game.isTerminal(tsp_state):
            pi = self.mcts.getActionProb(tsp_state, temp=1)
            # print(pi)
            trajectory.append((tsp_state, pi))

            action = np.random.choice(len(pi), p=pi)
            tsp_state = self.game.getNextState(tsp_state, action)

        # Terminal state reached
        final_tour_length = self.game.getTourLength(tsp_state)
        raw_value = (self.baseline - final_tour_length) / self.baseline
        value = np.clip(raw_value, -1, 1)
        # print(value)

        new_trainExamples = []
        for st, pi in trajectory:
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

            if (
                len(self.trainExamplesHistory)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning("Removing oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples()

            # Shuffle examples
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.old_net.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )

            pmcts = MCTS(self.game, self.old_net, self.args)
            print("examples from episode", len(trainExamples))
            final_pi_loss, final_v_loss = self.nnet.train(trainExamples)
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
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )

                # If this is a truly better solution than any found before (including the currently known best_tour_length),
                # update baseline.
                if (self.best_tour_length is None) or (best_so_far.current_length < self.best_tour_length):
                    self.best_tour_length = best_so_far.current_length
                    # Update the baseline to something around the best known solution, for instance equal to that new best_tour_length
                    self.args.L_baseline = self.best_tour_length

                    log.info(f"New best length found: {self.best_tour_length}. Updated baseline to this value.")

            else:
                best_so_far = best_state_old
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )


            # Write current iteration's losses and lengths to CSV
            try:
                with open(self.losses_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    # We'll record iteration-level data
                    writer.writerow(
                        [
                            i,
                            "",
                            "",
                            final_pi_loss,
                            final_v_loss,
                            self.avg_lengths_new[-1],
                            (
                                self.avg_lengths_old[-1]
                                if len(self.avg_lengths_old) == i
                                else ""
                            ),
                        ]
                    )
            except:
                log.error("losses file was in use, couldnt write to file.")

            # Update loss and length plot
            self.plot_loss_and_length_history()
            self.plot_value_loss_history()

            if self.args.visualize:
                rounded_len = round(best_so_far.current_length, 4)
                tour_str = "_".join(map(str, best_so_far.tour))
                title = f"Iter: {i} - Length: {rounded_len}"
                save_path = os.path.join(
                    self.folder,
                    "graphs",
                    f"Iter_{i:04}_len_{rounded_len:05}_{tour_str}.png",
                )
                self.game.plotTour(best_so_far, title=title, save_path=save_path)

        log.info("Average Tour Lengths per Iteration:")
        for iteration, (old_len, new_len) in enumerate(
            zip(self.avg_lengths_old, self.avg_lengths_new), 1
        ):
            log.info(
                f"Iteration {iteration}: Old Avg Length = {old_len}, New Avg Length = {new_len}"
            )

    def plot_loss_and_length_history(self):
        """
        Plot losses (pi and value) and tour lengths over iterations.
        We have:
        - iteration_pi_loss_history
        - iteration_v_loss_history
        - avg_lengths_new (new network)
        - avg_lengths_old (old network)
        - L_baseline: a fixed baseline line
        - nn_length: nearest neighbor solution line (purple)
        - best_tour_length: if not None, plot as another line
        """

        if len(self.iteration_pi_loss_history) == 0:
            return

        iterations = np.arange(1, len(self.iteration_pi_loss_history) + 1)

        plt.figure()
        # Left axis for losses
        ax1 = plt.gca()
        ax1.plot(
            iterations,
            self.iteration_pi_loss_history,
            label="Policy Loss",
            color="blue",
        )
        ax1.plot(
            iterations,
            self.iteration_v_loss_history,
            label="Value Loss",
            color="orange",
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        ax1.legend(loc="upper left")

        # Right axis for lengths
        ax2 = ax1.twinx()
        ax2.plot(
            iterations,
            self.avg_lengths_new,
            label="New Network Tour Length",
            color="green",
        )

        # Horizontal lines for references
        L_baseline = self.args.L_baseline
        ax2.axhline(
            y=L_baseline,
            color="red",
            linestyle="--",
            label="Baseline (Simple)",
        )
        # Nearest neighbor line
        ax2.axhline(
            y=self.nn_length,
            color="purple",
            linestyle="-.",
            label="Nearest Neighbor"
        )
        # If best_tour_length is known, plot it too
        if self.best_tour_length is not None:
            ax2.axhline(
                y=self.best_tour_length,
                color="brown",
                linestyle=":",
                label="Loaded Best Solution"
            )

        ax2.set_ylabel("Tour Length")
        ax2.legend(loc="upper right")

        plt.title("Policy/Value Losses and Tour Length per Iteration")

        loss_plot_path = os.path.join(
            self.folder, "graphs", "loss_and_length_history.png"
        )
        plt.savefig(loss_plot_path)
        plt.close()

    def plot_value_loss_history(self):
        """
        Plot only the value loss over iterations separately,
        since value loss might be smaller and we want a better scale.
        """
        if len(self.iteration_v_loss_history) == 0:
            return

        iterations = np.arange(1, len(self.iteration_v_loss_history) + 1)

        plt.figure()
        plt.plot(iterations, self.iteration_v_loss_history, color='orange', label="Value Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Value Loss")
        plt.title("Value Loss Over Iterations")
        plt.grid(True)
        plt.legend()

        value_loss_path = os.path.join(self.folder, "graphs", "value_loss_history.png")
        plt.savefig(value_loss_path)
        plt.close()

    def evaluateNetwork(self, mcts: MCTS, name=""):
        tsp_state = self.game.getInitState()
        best_tsp_state = tsp_state

        for _ in range(self.game.getActionSize()):
            # Check terminal condition
            if self.game.isTerminal(tsp_state):
                break

            pi = mcts.getActionProb(tsp_state, temp=0)

            # Mask out invalid moves again just in case
            valid_moves = self.game.getValidMoves(tsp_state)
            pi = pi * valid_moves
            if np.sum(pi) == 0:
                # No valid moves remain
                break
            pi = pi / np.sum(pi)

            action = np.argmax(pi)

            next_tsp_state = self.game.getNextState(tsp_state, action)
            if next_tsp_state.current_length > best_tsp_state.current_length:
                best_tsp_state = next_tsp_state
            tsp_state = next_tsp_state

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

    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "checkpoint.examples")
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
