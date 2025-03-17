import logging
import os
import csv
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from MCTS import MCTS
import NNetWrapper
from TSPGame import TSPGame
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
        print(f"Best tour length: {self.best_tour_length}")

        # For stable evaluation
        self.coords_for_eval = coords_for_eval or []
        self.nn_lengths_for_eval = nn_lengths_for_eval or []

        # We'll store final length per iteration *per evaluation set* in a 2D structure:
        # shape (n_eval_sets, n_iterations)
        self.eval_set_lengths_history = (
            []
        )  # Will append an array of size n_eval_sets each iteration.

        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []

        # Track best average length across stable sets
        self.best_avg_length = float("inf")

        # Logging for losses
        self.iteration_pi_loss_history = []
        self.iteration_v_loss_history = []
        self.eval_avg_length_history = []

        # We'll store the large chart of policy/value losses in the main run folder
        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"]
            )

        # Optionally run a pre-training eval
        self.preTrainingEval()

    ###############################################################################
    # 1) Updated 'preTrainingEval' method to match the same folder structure
    #    and naming conventions as after training.
    ###############################################################################
    def preTrainingEval(self):
        """
        Evaluate the untrained model on *all* coords_for_eval sets if provided,
        storing each set's final route in 'tours' or 'tours/other sets'.
        Set #1 => tours\iter_000_len_{XXXX}.png
        Others => tours\other sets\set{k}_iter_000_len_{XXXX}.png
        """
        log.info("=== Pre-Training Evaluation ===")
        if not self.coords_for_eval:
            log.info("No coords_for_eval => skipping pretrain eval.")
            return

        # We'll keep iteration=0 for the naming:
        iteration_str = "000"

        # Create needed folders
        tours_folder = os.path.join(self.folder, "tours")
        os.makedirs(tours_folder, exist_ok=True)
        other_sets_folder = os.path.join(tours_folder, "other sets")
        os.makedirs(other_sets_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        total_len = 0.0
        N = len(self.coords_for_eval)

        for idx, coords in enumerate(self.coords_for_eval):
            # Evaluate from start_node=0
            self.game.node_coordinates = coords
            state = self.game.getInitState()
            temp_mcts = MCTS(self.game, self.nnet, self.args)

            while not self.game.isTerminal(state):
                pi = temp_mcts.getActionProb(state, temp=0)
                pi = np.array(pi, dtype=float)
                pi *= self.game.getValidMoves(state)
                if pi.sum() < 1e-12:
                    break
                pi /= pi.sum()
                action = np.argmax(pi)
                state = self.game.getNextState(state, action)

            length = state.current_length
            total_len += length
            log.info(f"[PreTrainEval] Set {idx+1}/{N}, length = {length:.4f}")

            # Save under the same naming logic as post-training:
            if idx == 0:
                # Set #1 => direct in 'tours'
                filename = f"iter_{iteration_str}_len_{length:.4f}.png"
                out_path = os.path.join(tours_folder, filename)
            else:
                # Other sets => 'tours/other sets'
                filename = f"set{idx+1}_iter_{iteration_str}_len_{length:.4f}.png"
                out_path = os.path.join(other_sets_folder, filename)

            self.game.plotTour(
                state,
                title=f"PreEval Set {idx+1} (Len={length:.4f})",
                save_path=out_path,
            )

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

        state = self.game.getInitState()
        trajectory = []

        while not self.game.isTerminal(state):
            pi = self.mcts.getActionProb(state, temp=1)
            trajectory.append((state, pi))
            action = np.random.choice(len(pi), p=pi)
            state = self.game.getNextState(state, action)

        final_len = self.game.getTourLength(state)
        examples = []
        for st, pi in trajectory:
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
                episode_data = (
                    self.executeEpisode()
                )  # list of (state, pi, leftover_dist)

                # *** AUGMENT BEFORE ADDING TO iterationTrainExamples ***
                if getattr(self.args, "augmentationFactor", 1) > 1:
                    augmented_data = self.augmentExamples(episode_data)
                    iterationTrainExamples.extend(augmented_data)
                else:
                    iterationTrainExamples.extend(episode_data)

            # Now add the iterationTrainExamples to trainExamplesHistory
            self.trainExamplesHistory.append(iterationTrainExamples)

            if (
                len(self.trainExamplesHistory)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning("Removing oldest entry in trainExamplesHistory.")
                self.trainExamplesHistory.pop(0)

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

            # Evaluate on stable coords, etc...
            eval_lens = self.evaluateAllCoords()
            avg_len = float(np.mean(eval_lens)) if eval_lens else float("inf")
            self.eval_avg_length_history.append(avg_len)
            self.eval_set_lengths_history.append(eval_lens)

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
            do_full_plot = hasattr(self.args, "plot_all_eval_sets_interval") and (
                i % self.args.plot_all_eval_sets_interval == 0
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

    def augmentExamples(self, original_data):
        """
        For each (TSPState, pi, leftover_dist), produce (args.augmentationFactor - 1)
        additional variations using random label permutations and rotation around (0.5, 0.5).

        :param original_data: list of (state, pi, leftover_val).
        :return: the new extended list.
        """
        factor = getattr(self.args, "augmentationFactor", 1)
        if factor <= 1:
            return original_data

        augmented = []
        for state, pi, leftover in original_data:
            # Always keep the original
            augmented.append((state, pi, leftover))

            for _ in range(factor - 1):
                # 1) Apply a random permutation
                permuted_state, permuted_pi = self.applyRandomPermutation(state, pi)
                # 2) Apply rotation about (0.5, 0.5)
                rotated_state, rotated_pi = self.applyCenterRotation(
                    permuted_state, permuted_pi
                )
                # leftover remains the same if distance truly unchanged
                augmented.append((rotated_state, rotated_pi, leftover))

        return augmented

    def applyRandomPermutation(self, state, pi):
        """
        Randomly permute labels [0..n-1].
        We create new TSPState, reorder node coords accordingly,
        reorder the partial tour, and fix the unvisited array.
        Also reorder the pi distribution.
        """
        n = state.num_nodes
        perm = np.random.permutation(n)  # e.g. [2,0,1,...]

        old_coords = np.array(state.node_coordinates)
        new_coords = old_coords[perm].tolist()

        old_tour = state.tour
        new_tour = [perm[node] for node in old_tour]

        old_unvisited = state.unvisited
        new_unvisited = np.zeros_like(old_unvisited)
        for old_lbl in range(n):
            if old_unvisited[old_lbl] == 1:
                new_lbl = perm[old_lbl]
                new_unvisited[new_lbl] = 1

        from TSPState import TSPState

        new_state = TSPState(n, new_coords)
        new_state.tour = new_tour
        new_state.unvisited = new_unvisited
        # if you recompute the partial cost, do:
        # new_state.current_length = self.recomputeTourLength(new_state)
        # otherwise copy:
        # new_state.current_length = state.current_length

        # reorder pi
        new_pi = np.zeros(n, dtype=float)
        for old_label, prob in enumerate(pi):
            new_label = perm[old_label]
            new_pi[new_label] = prob

        return new_state, new_pi

    def applyCenterRotation(self, state, pi):
        """
        Rotate all coordinates about (0.5, 0.5) by a random angle in [0, 2*pi).
        Distances remain the same if TSP is in [0,1]^2.
        We keep the same node labeling (tour/unvisited).
        pi does not need label reorder, just the same array.

        If you want partial cost to remain identical,
        either recalc or trust that the TSP code uses purely index-based cost
        => same leftover is valid if everything in [0,1]^2 doesn't break distance.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        cosA, sinA = np.cos(angle), np.sin(angle)

        coords = state.node_coordinates
        rotated_coords = []
        for x, y in coords:
            # shift center to (0.0,0.0)
            dx = x - 0.5
            dy = y - 0.5
            # rotate
            rx = dx * cosA - dy * sinA
            ry = dx * sinA + dy * cosA
            # shift back
            rx += 0.5
            ry += 0.5
            rotated_coords.append([rx, ry])

        from TSPState import TSPState

        new_state = TSPState(state.num_nodes, rotated_coords)
        new_state.tour = list(state.tour)
        new_state.unvisited = state.unvisited.copy()
        # new_state.current_length = state.current_length
        # or if TSP code automatically re-checks distances, do:
        # new_state.current_length = self.recomputeTourLength(new_state)

        return new_state, np.array(pi, copy=True)

    def evaluateAllCoords(self):
        """
        Evaluate the *current* net on the entire coords_for_eval list.
        Return a list of final lengths, one per eval set.
        """
        if not self.coords_for_eval:
            log.info(
                "No coords_for_eval => skipping stable evaluation => returning empty list"
            )
            return []

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        lengths = []
        for _, coords in enumerate(self.coords_for_eval):
            self.game.node_coordinates = coords
            state = self.game.getInitState()
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

    ###############################################################################
    # 2) Where you used 'plotAllEvalTours', rename folder to 'tours/other sets'
    #    Also note we only do this every X iterations and skip set #1 if you desire.
    ###############################################################################
    def plotAllEvalTours(self, iteration):
        """
        Evaluate + plot final routes for *all* stable coords in subplots or individually,
        but saving them in 'tours/other sets', named 'set{k}_iter_{iteration:03d}_len_{X}.png'.
        Typically called only every 'plot_all_eval_sets_interval' iteration.
        """
        log.info(f"Plotting all stable tours for iteration {iteration} ...")

        # The 'other sets' folder
        other_folder = os.path.join(self.folder, "tours", "other sets")
        os.makedirs(other_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        # If you want to skip set #1 here (since you're plotting it every iteration),
        # you can do coords_for_eval[1:] below. But if you want all sets, use entire list:
        for idx, coords in enumerate(self.coords_for_eval):
            # If you'd like to skip the first set in 'other sets', do:
            # if idx == 0: continue  # skip set1

            self.game.node_coordinates = coords
            state = self.game.getInitState()
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
            out_path = os.path.join(other_folder, filename)
            self.game.plotTour(
                state,
                title=f"EvalSet {idx+1}, Iter={iteration}, Len={length:.4f}",
                save_path=out_path,
            )

        self.args.numMCTSSims = original_sims

    ###############################################################################
    # 1) Where you used "evaluation" folder, rename it to "tours" and store only set #1
    ###############################################################################
    def plotSingleEvalTour(self, coords, iteration, set_idx=1):
        """
        Plots the final route for coordinate set #1 each iteration.
        Output file in 'tours/' => 'iter_{iteration:03d}_len_{length:.4f}.png'
        """
        tours_folder = os.path.join(self.folder, "tours")
        os.makedirs(tours_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        self.game.node_coordinates = coords
        state = self.game.getInitState()
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
        filename = f"iter_{iteration:03d}_len_{length:.4f}.png"
        out_path = os.path.join(tours_folder, filename)

        self.game.plotTour(
            state, title=f"Set1, Iter={iteration}, Len={length:.4f}", save_path=out_path
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
        axL.plot(
            iters,
            self.eval_avg_length_history,
            label="Current Length",
            color="blue",
            linewidth=2,
        )

        # Add horizontal line for the optimal solution length if available
        if self.best_tour_length is not None:
            axL.axhline(
                y=self.best_tour_length,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Optimal Solution ({self.best_tour_length:.1f})",
                zorder=5,  # Make sure optimal line is on top
            )

            # Set y-axis limits to ensure optimal line is visible
            if self.eval_avg_length_history:
                # Get the full range of values we need to show
                all_values = self.eval_avg_length_history + [self.best_tour_length]
                max_length = max(all_values)
                min_length = min(all_values)

                # Calculate range and add padding
                value_range = max_length - min_length
                padding = value_range * 0.2  # 20% padding

                # Ensure minimum padding of 10% of optimal length
                min_padding = self.best_tour_length * 0.1
                padding = max(padding, min_padding)

                y_min = min_length - padding
                y_max = max_length + padding
            else:
                # If no data points yet, center around optimal with generous padding
                padding = self.best_tour_length * 0.5  # 50% padding when no data
                y_min = self.best_tour_length - padding
                y_max = self.best_tour_length + padding

            axL.set_ylim(y_min, y_max)

        axL.set_xlabel("Iteration")
        axL.set_ylabel("Tour Length")
        axL.set_title("Tour Length Evolution")
        axL.grid(True, alpha=0.3)
        axL.legend(loc="upper right")

        # Right: policy + value loss
        ax1 = axR
        ax1.plot(
            iters, self.iteration_pi_loss_history, label="Policy Loss", color="blue"
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Policy Loss", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(
            iters, self.iteration_v_loss_history, label="Value Loss", color="orange"
        )
        ax2.set_ylabel("Value Loss", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.set_yscale("log")

        # If we have enough data, skip the first 5 from the range
        if len(self.iteration_v_loss_history) > 5:
            tail_vals = self.iteration_v_loss_history[5:]
            min_tail = min(tail_vals)
            max_tail = max(tail_vals)
            ax2.set_ylim(min_tail * 0.8, max_tail * 1.2)

        ax2.legend(loc="upper right")

        # Add current performance to title
        if self.best_tour_length is not None:
            gap = ((eval_len - self.best_tour_length) / self.best_tour_length) * 100
            title = f"Iter {iteration}: Length={eval_len:.1f} (Gap: {gap:.1f}%)"
        else:
            title = f"Iter {iteration}: Length={eval_len:.1f}"

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()

        # Save in main folder with higher DPI for better quality
        loss_plot_path = os.path.join(self.folder, f"loss_and_length_history.png")
        fig.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    ###############################################################################
    # 2) In 'plotMultiEvalSubplots', ensure we do up to 4 columns AND
    #    also draw a nearest neighbor line for each set if self.nn_lengths_for_eval is provided.
    ###############################################################################
    def plotMultiEvalSubplots(self, iteration):
        """
        Multi-subplot figure with up to 4 columns, one subplot per evaluation set.
        We'll plot the final length across all iterations for that set,
        plus a horizontal line for the NN solution if provided.
        We'll store the figure in the main folder named 'evaluation_subplots.png'.
        """
        if not self.coords_for_eval:
            return

        n_sets = len(self.coords_for_eval)
        # x-axis: iterations from 1..(current iteration index)
        x = np.arange(1, len(self.eval_set_lengths_history) + 1)

        # Up to 4 columns
        cols = 4
        rows = (n_sets + cols - 1) // cols  # ceiling division

        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
        )
        axes = axes.flatten()  # flatten so we can index easily

        for idx in range(n_sets):
            lengths_for_this_set = [hist[idx] for hist in self.eval_set_lengths_history]
            ax = axes[idx]
            ax.plot(x, lengths_for_this_set, marker="o", label=f"EvalSet {idx+1}")

            # if we have NN length
            if idx < len(self.nn_lengths_for_eval):
                nn_len = self.nn_lengths_for_eval[idx]
                ax.axhline(y=nn_len, color="red", linestyle="--", label="NN Length")

            ax.set_ylabel("Tour Length")
            ax.set_title(f"Set {idx+1} Over Iterations")
            ax.grid(True)
            ax.legend()

        # Hide leftover axes
        for i in range(n_sets, rows * cols):
            axes[i].set_visible(False)

        # Label X-axis
        for ax in axes:
            ax.set_xlabel("Iteration")

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

            os.makedirs(folder)
        filename = os.path.join(folder, "checkpoint.examples")
        with open(filename, "wb") as f:
            Pickler(f).dump(self.trainExamplesHistory)
