import logging
import os
import random
import csv
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from MCTS import MCTS
from TSP.pytorch.NNetWrapper import NNetWrapper
from TSP.TSPGame import TSPGame
from utils import *

log = logging.getLogger(__name__)

class Coach:
    def __init__(
        self, 
        game: TSPGame,
        nnet: NNetWrapper,
        args,
        best_tour_length=None,
        folder=None,
        coords_for_eval=None,
        nn_lengths_for_eval=None,
    ):
        """
        :param coords_for_eval: A list of coordinate sets for stable evaluation.
        :param nn_lengths_for_eval: If you have nearest-neighbor solutions for each coords_for_eval,
                                    pass them in a list of the same length. We'll use them as horizontal lines
                                    in the multi-subplot chart. If not, pass None or an empty list.
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.folder = folder or "."

        # Possibly known best from TSPLIB
        self.best_tour_length = best_tour_length

        # For stable evaluation
        self.coords_for_eval = coords_for_eval or []
        self.nn_lengths_for_eval = nn_lengths_for_eval or []

        # We'll store final length per iteration *per evaluation set* in a 2D structure:
        # shape (n_eval_sets, n_iterations)
        self.eval_set_lengths_history = []  # Will append an array of size n_eval_sets each iteration.

        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []

        # Track best average length across stable sets
        self.best_avg_length = float('inf')

        # Logging for losses
        self.iteration_pi_loss_history = []
        self.iteration_v_loss_history = []
        self.eval_avg_length_history = []

        # We'll store the large chart of policy/value losses in the main run folder
        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"])

        # Optionally run a pre-training eval
        self.preTrainingEval()

    def preTrainingEval(self):
        """
        Evaluate the untrained model on *all* coords_for_eval sets if provided.
        We'll name them "iter_000_len_XXXX.png" for the Pretrain stage (iteration=0).
        We'll store them in the main folder or a subfolder "evaluation".
        """
        log.info("=== Pre-Training Evaluation ===")
        if not self.coords_for_eval:
            log.info("No coords_for_eval => skipping pretrain eval.")
            return

        # We'll create a subfolder named "evaluation" for these pretrain tours
        eval_folder = os.path.join(self.folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        total_len = 0.0
        N = len(self.coords_for_eval)

        for idx, coords in enumerate(self.coords_for_eval):
            self.game.node_coordinates = coords
            # Evaluate from start_node=0
            state = self.game.getInitEvalState(start_node=0)
            temp_mcts = MCTS(self.game, self.nnet, self.args)

            while not self.game.isTerminal(state):
                pi = temp_mcts.getActionProb(state, temp=0)
                pi = np.array(pi, dtype=float)
                pi *= self.game.getValidMoves(state)
                sum_pi = np.sum(pi)
                if sum_pi < 1e-12:
                    break
                pi /= sum_pi

                action = np.argmax(pi)
                state = self.game.getNextState(state, action)

            length = state.current_length
            total_len += length
            log.info(f"[PreTrainEval] EvalSet {idx+1}/{N}, length = {length:.4f}")

            # Use iteration=0 in the filename
            filename = f"set{idx+1}_iter_000_len_{length:.4f}.png"
            savepath = os.path.join(eval_folder, filename)
            self.game.plotTour(state, title=f"PreEval Set {idx+1} (Len={length:.4f})", save_path=savepath)

        # Average
        avg_len = total_len / N
        log.info(f"Avg length across {N} eval sets in Pre-Training: {avg_len:.4f}")
        self.args.numMCTSSims = original_sims

    def executeEpisode(self):
        """
        Self-play with random coords if we are not reading from file => Overwrite the game coords.
        Then do MCTS from random start => leftover distance as target value.
        """
        if not self.args.read_from_file:
            new_coords = np.random.rand(self.game.num_nodes, 2).tolist()
            self.game.node_coordinates = new_coords

        state = self.game.getInitState(randomize_start=True)
        trajectory = []

        while not self.game.isTerminal(state):
            pi = self.mcts.getActionProb(state, temp=1)
            trajectory.append((state, pi))
            action = np.random.choice(len(pi), p=pi)
            state = self.game.getNextState(state, action)

        final_len = self.game.getTourLength(state)
        examples = []
        for (st, pi) in trajectory:
            leftover = final_len - st.current_length
            examples.append((st, pi, leftover))

        return examples

    def learn(self):
        for i in range(1, self.args.numIters + 1):
            log.info(f"=== Starting Iter #{i} ===")
            iterationTrainExamples = []

            # Self-play
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.game, self.nnet, self.args)
                iterationTrainExamples.extend(self.executeEpisode())

            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("Removing oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples()

            # Flatten & shuffle
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            log.info(f"Number of training examples: {len(trainExamples)}")
            self.nnet.save_checkpoint(self.args.checkpoint, "temp.pth.tar")

            pi_loss, v_loss = self.nnet.train(trainExamples)
            self.iteration_pi_loss_history.append(pi_loss)
            self.iteration_v_loss_history.append(v_loss)

            # Evaluate on stable coords
            eval_lens = self.evaluateAllCoords()
            # eval_lens is e.g. a list of final lengths [len_set1, len_set2, ...]
            # average them
            avg_len = float(np.mean(eval_lens))
            self.eval_avg_length_history.append(avg_len)

            # Store these set-level lengths for multi-subplot
            self.eval_set_lengths_history.append(eval_lens)

            log.info(f"Avg length on stable eval sets: {avg_len:.4f}")

            # If improved => store as best, else keep going
            if avg_len <= self.best_avg_length:
                self.best_avg_length = avg_len
                log.info("New best average => saving model as best.pth.tar")
                self.nnet.save_checkpoint(self.args.checkpoint, "best.pth.tar")
            else:
                log.info("No improvement => continuing")

            # Plot one or all stable sets?
            # If you want to plot all sets every X iterations
            do_full_plot = (
                hasattr(self.args, "plot_all_eval_sets_interval") and
                (i % self.args.plot_all_eval_sets_interval == 0)
            )
            if do_full_plot and self.coords_for_eval:
                self.plotAllEvalTours(i)
            else:
                # Just plot the FIRST stable coord set's resulting tour
                if self.coords_for_eval:
                    self.plotSingleEvalTour(
                        coords=self.coords_for_eval[0],
                        iteration=i,
                        set_idx=1,
                    )

            # Plot overall losses & single-line average
            self.plot_loss_and_length_history(i, avg_len)
            # Also plot the multi-subplot figure for each evaluation set
            self.plotMultiEvalSubplots(i)

    def evaluateAllCoords(self):
        """
        Evaluate the *current* net on the entire coords_for_eval list.
        Return a list of final lengths, one per eval set.
        """
        if not self.coords_for_eval:
            log.info("No coords_for_eval => skipping stable evaluation => returning empty list")
            return []

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        lengths = []
        for idx, coords in enumerate(self.coords_for_eval):
            self.game.node_coordinates = coords
            state = self.game.getInitEvalState(0)
            temp_mcts = MCTS(self.game, self.nnet, self.args)

            while not self.game.isTerminal(state):
                pi = temp_mcts.getActionProb(state, temp=0)
                pi = np.array(pi, dtype=float)
                pi *= self.game.getValidMoves(state)
                sum_pi = np.sum(pi)
                if sum_pi < 1e-12:
                    break
                pi /= sum_pi

                action = np.argmax(pi)
                state = self.game.getNextState(state, action)

            length = state.current_length
            lengths.append(length)

        self.args.numMCTSSims = original_sims
        return lengths

    def plotAllEvalTours(self, iteration):
        """
        Plots final tours for *all* coords_for_eval from start_node=0
        in a subfolder "evaluation", named "set{idx+1}_iter_010_len_3.2342.png"
        """
        log.info(f"Plotting all stable tours for iteration {iteration} ...")

        eval_folder = os.path.join(self.folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        for idx, coords in enumerate(self.coords_for_eval):
            self.game.node_coordinates = coords
            state = self.game.getInitEvalState(0)
            temp_mcts = MCTS(self.game, self.nnet, self.args)

            while not self.game.isTerminal(state):
                pi = temp_mcts.getActionProb(state, temp=0)
                pi = np.array(pi, dtype=float)
                pi *= self.game.getValidMoves(state)
                sum_pi = np.sum(pi)
                if sum_pi < 1e-12:
                    break
                pi /= sum_pi
                action = np.argmax(pi)
                state = self.game.getNextState(state, action)

            length = state.current_length
            filename = f"set{idx+1}_iter_{iteration:03d}_len_{length:.4f}.png"
            path = os.path.join(eval_folder, filename)
            self.game.plotTour(
                state,
                title=f"EvalSet {idx+1}, Iter={iteration}, Len={length:.4f}",
                save_path=path
            )

        self.args.numMCTSSims = original_sims

    def plotSingleEvalTour(self, coords, iteration, set_idx=1):
        """
        Plots the final route for a single coordinate set (e.g. the first set),
        naming it "set{set_idx}_iter_###_len_XXXX.png" in the "evaluation" folder.
        """
        eval_folder = os.path.join(self.folder, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        self.game.node_coordinates = coords
        state = self.game.getInitEvalState(0)
        temp_mcts = MCTS(self.game, self.nnet, self.args)

        while not self.game.isTerminal(state):
            pi = temp_mcts.getActionProb(state, temp=0)
            pi = np.array(pi, dtype=float)
            pi *= self.game.getValidMoves(state)
            sum_pi = np.sum(pi)
            if sum_pi < 1e-12:
                break
            pi /= sum_pi

            action = np.argmax(pi)
            state = self.game.getNextState(state, action)

        length = state.current_length
        filename = f"set{set_idx}_iter_{iteration:03d}_len_{length:.4f}.png"
        path = os.path.join(eval_folder, filename)
        self.game.plotTour(
            state,
            title=f"EvalSet {set_idx}, Iter={iteration}, Len={length:.4f}",
            save_path=path
        )

        self.args.numMCTSSims = original_sims

    def plot_loss_and_length_history(self, iteration, eval_len):
        """
        We'll store the chart in the main folder (not in "graphs").
        We'll skip the first 5 points from the Value Loss axis using set_ylim if we have enough points.
        """
        iters = range(1, len(self.iteration_pi_loss_history) + 1)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6))

        # Left: stable eval length over iterations
        axL.plot(iters, self.eval_avg_length_history, label='Eval Avg Length', color='green')
        axL.set_xlabel('Iteration')
        axL.set_ylabel('Final Tour Length')
        axL.set_title('Eval Tour Length (Stable) Over Iterations')
        axL.legend()

        # Right: policy + value loss
        ax1 = axR
        ax1.plot(iters, self.iteration_pi_loss_history, label='Policy Loss', color='blue')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Policy Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(iters, self.iteration_v_loss_history, label='Value Loss', color='orange')
        ax2.set_ylabel('Value Loss', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_yscale('log')

        # If we have enough data, skip the first 5 from the range
        if len(self.iteration_v_loss_history) > 5:
            # e.g. find the min of Value Loss after the first 5
            tail_vals = self.iteration_v_loss_history[5:]
            min_tail = min(tail_vals)
            max_tail = max(tail_vals)
            ax2.set_ylim(min_tail * 0.8, max_tail * 1.2)  # some margin

        ax2.legend(loc='upper right')

        fig.suptitle(f"Iter {iteration}: EvalLen={eval_len:.4f}", fontsize=14)
        fig.tight_layout()

        # Save in main folder
        loss_plot_path = os.path.join(self.folder, f"loss_and_length_history.png")
        fig.savefig(loss_plot_path)
        plt.close(fig)

    def plotMultiEvalSubplots(self, iteration):
        """
        Creates a multi-subplot figure with one subplot for each evaluation set.
        We'll plot the final length across all iterations for that set,
        plus a horizontal line for the NN solution if provided.
        We'll store the figure in the main folder named "evaluation_subplots.png".
        """
        if not self.coords_for_eval:
            return

        n_sets = len(self.coords_for_eval)
        # Ensure we have an array of shape (#iterations, #eval_sets)
        # => we stored each iteration as a 1D array of size n_sets in self.eval_set_lengths_history

        # The length is #iteration => x-axis
        # For each set, we have a line over iteration
        x = np.arange(1, len(self.eval_set_lengths_history) + 1)

        fig, axs = plt.subplots(n_sets, 1, figsize=(7, 4 * n_sets), sharex=True)
        if n_sets == 1:
            axs = [axs]  # Make it iterable

        for idx in range(n_sets):
            # gather the length for set idx across all iteration
            lengths_for_this_set = [hist[idx] for hist in self.eval_set_lengths_history]
            ax = axs[idx]
            ax.plot(x, lengths_for_this_set, marker='o', label=f"EvalSet {idx+1}")
            # if we have a nearest neighbor length for it:
            if idx < len(self.nn_lengths_for_eval):
                nn_len = self.nn_lengths_for_eval[idx]
                ax.axhline(y=nn_len, color='red', linestyle='--', label='NN Length')

            ax.set_ylabel("Tour Length")
            ax.set_title(f"Evaluation Set {idx+1} Over Iterations")
            ax.grid(True)
            ax.legend()

        axs[-1].set_xlabel("Iteration")
        fig.suptitle(f"Multiple Evaluation Sets Subplots (Iter={iteration})")

        out_path = os.path.join(self.folder, "evaluation_subplots.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def saveTrainExamples(self):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "checkpoint.examples")
        with open(filename, "wb") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        folder, fname = self.args.load_folder_file
        modelFile = os.path.join(folder, fname)
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Pickler.load(f)
            log.info("Loading done!")
