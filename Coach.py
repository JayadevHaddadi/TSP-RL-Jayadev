import logging
import os
import random
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
    def __init__(self, game: TSPGame, nnet: NNetWrapper, args, best_tour_length=None, folder=None):
        self.game = game
        self.nnet = nnet
        self.old_net = nnet.__class__(game, args)
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []
        self.best_tour_length = best_tour_length
        self.folder = folder

        self.iteration_pi_loss_history = []
        self.iteration_v_loss_history = []
        self.avg_lengths_new = []
        self.avg_lengths_old = []

        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"])

    def executeEpisode(self):
        # randomize_start=True so each training run starts from a random node
        tsp_state = self.game.getInitState(randomize_start=True)
        trajectory = []
        while not self.game.isTerminal(tsp_state):
            pi = self.mcts.getActionProb(tsp_state, temp=1)
            trajectory.append((tsp_state, pi))
            action = np.random.choice(len(pi), p=pi)
            tsp_state = self.game.getNextState(tsp_state, action)
        final_tour_length = self.game.getTourLength(tsp_state)
        new_trainExamples = []
        for (st, pi) in trajectory:
            leftover_dist = final_tour_length - st.current_length
            new_trainExamples.append((st, pi, leftover_dist))
        return new_trainExamples


    def learn(self):
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")
            iterationTrainExamples = []

            # --- Self-play to gather data ---
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)
                iterationTrainExamples.extend(self.executeEpisode())

            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("Removing oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples()

            # Shuffle and train new network
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")
            self.old_net.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

            old_mcts = MCTS(self.game, self.old_net, self.args)
            log.info("examples from episode " + str(len(trainExamples)))
            final_pi_loss, final_v_loss = self.nnet.train(trainExamples)
            new_mcts = MCTS(self.game, self.nnet, self.args)

            # Record final losses for this iteration
            self.iteration_pi_loss_history.append(final_pi_loss)
            self.iteration_v_loss_history.append(final_v_loss)

            # --- Randomly choose a single start node for evaluation ---
            eval_start_node = random.randint(0, self.game.num_nodes - 1)

            log.info("EVALUATING NEW NETWORK from start node %d", eval_start_node)
            best_state_old = self.evaluateNetwork(old_mcts, start_node=eval_start_node, name="Old")
            best_state_new = self.evaluateNetwork(new_mcts, start_node=eval_start_node, name="New")

            self.avg_lengths_old.append(best_state_old.current_length)
            self.avg_lengths_new.append(best_state_new.current_length)

            log.info(
                f"Average Tour Length - New: {best_state_new.current_length}, "
                f"Old: {best_state_old.current_length}"
            )

            if best_state_new.current_length <= best_state_old.current_length:
                best_so_far = best_state_new
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename="best.pth.tar")

                if (self.best_tour_length is None) or (best_so_far.current_length < self.best_tour_length):
                    self.best_tour_length = best_so_far.current_length
                    self.args.L_baseline = self.best_tour_length
                    log.info(f"New best length found: {self.best_tour_length}. Updated baseline to this value.")
            else:
                best_so_far = best_state_old
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename="temp.pth.tar")

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
        if len(self.avg_lengths_new) == 0 or len(self.avg_lengths_old) == 0:
            return
        
        iterations = np.arange(1, len(self.avg_lengths_new) + 1)

        plt.figure(figsize=(12, 6))

        # --- Left Plot: Average Tour Lengths ---
        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.avg_lengths_new, label='New Network Tour Length', color='green')
        plt.plot(iterations, self.avg_lengths_old, label='Old Network Tour Length', color='blue')

        # Optionally plot your nearest neighbor solution or best known solution as horizontal lines
        plt.axhline(y=self.args.NN_length, color='purple', linestyle='-.', label='Nearest Neighbor')
        if self.best_tour_length is not None:
            plt.axhline(y=self.best_tour_length, color='brown', linestyle=':', label='Loaded Best Solution')

        plt.xlabel('Iteration')
        plt.ylabel('Tour Length')
        plt.title('Average Tour Lengths per Iteration')
        plt.legend()

        # --- Right Plot: Policy and Value Losses ---
        plt.subplot(1, 2, 2)
        ax1 = plt.gca()
        ax1.plot(iterations, self.iteration_pi_loss_history, label='Policy Loss', color='blue')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Policy Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        # Create a secondary Y-axis for value loss
        ax2 = ax1.twinx()
        ax2.plot(iterations, self.iteration_v_loss_history, label='Value Loss', color='orange')
        ax2.set_ylabel('Value Loss', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        # **Key line to enable log scale for value loss:**
        ax2.set_yscale('log')

        ax2.legend(loc='upper right')
        plt.title('Policy and Value Losses per Iteration')

        plt.tight_layout()
        
        loss_plot_path = os.path.join(self.folder, 'loss_and_length_history.png')
        plt.savefig(loss_plot_path)
        plt.close()


    def evaluateNetwork(self, mcts: MCTS, start_node=0, name=""):
        """
        Evaluate the network from a given start node so old and new networks
        get exactly the same test environment.
        """
        tsp_state = self.game.getInitEvalState(start_node)
        for _ in range(self.game.getActionSize()):
            if self.game.isTerminal(tsp_state):
                break
            pi = mcts.getActionProb(tsp_state, temp=0)
            valid_moves = self.game.getValidMoves(tsp_state)
            pi = pi * valid_moves
            if np.sum(pi) == 0:
                break
            pi = pi / np.sum(pi)
            action = np.argmax(pi)
            next_tsp_state = self.game.getNextState(tsp_state, action)
            tsp_state = next_tsp_state
        current_length = tsp_state.current_length
        best_tour_length = self.best_tour_length
        if best_tour_length is not None:
            log.info(f"Best Known Tour Length: {best_tour_length}")
            log.info(f"Current best Length for {name}: {current_length}")
            improvement = ((current_length - best_tour_length) / best_tour_length) * 100
            log.info(f"Percentage above best known: {improvement:.2f}%")
        else:
            log.info(f"Current best Length for {name}: {current_length}")
        return tsp_state

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
