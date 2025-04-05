import logging
import os
import csv
import copy
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
        # Instead of copying the network, we'll create a new instance with the same architecture
        self.champion_nnet = None  # Will be initialized in learn()
        self.args = args
        self.folder = folder or "."

        # Possibly known best from TSPLIB
        self.best_tour_length = best_tour_length
        log.info(f"Best tour length: {self.best_tour_length}")

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

        # Track accept/reject decisions
        self.accepted_iterations = []
        self.rejected_iterations = []

        # We'll store the large chart of policy/value losses in the main run folder
        self.losses_file = os.path.join(self.folder, "losses.csv")
        with open(self.losses_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Iteration", "Epoch", "Batch", "Policy Loss", "Value Loss"]
            )

        # Add tracking for best tour lengths for each evaluation set
        self.best_eval_lengths = (
            [float("inf")] * len(self.coords_for_eval) if self.coords_for_eval else []
        )

        # Add tracking for best tour found during any episode
        self.best_episode_tour_length = float("inf")
        self.best_episode_tour_iteration = None
        self.best_episode_tour_episode = None
        self.best_episode_tour_history = []  # For plotting progress over time

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
            # Evaluate from configured start node
            start_node = (
                self.args.get("fixed_start_node", 0)
                if self.args.get("fixed_start", True)
                else np.random.choice(len(coords))
            )
            self.game.node_coordinates = coords
            state = self.game.getInitState(start_node=start_node)
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

            # Always save pre-training evaluations (these are the initial best lengths)
            self.best_eval_lengths[idx] = length

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
        Then do MCTS from configured start node => leftover distance as target value.
        """
        if not self.args.tsp_instance:
            new_coords = np.random.rand(self.game.num_nodes, 2).tolist()
            self.game.node_coordinates = new_coords

            # Reset distance matrix in game after changing coordinates
            self.game.distance_matrix = self.game._compute_distance_matrix()

        # Modified start node selection
        if self.args.get("fixed_start", True):
            start_node = self.args.get("fixed_start_node", 0)
        else:
            start_node = np.random.choice(self.game.num_nodes)

        state = self.game.getInitState(start_node=start_node)
        trajectory = []

        # Cache string representations to avoid recomputation
        visited_states = set()

        while not self.game.isTerminal(state):
            # Get string representation once per state
            state_str = self.game.uniqueStringRepresentation(state)

            # Track visited states for analysis
            visited_states.add(state_str)

            # Get action probabilities - reusing existing tree knowledge
            pi = self.mcts.getActionProb(state, temp=1)
            trajectory.append((state, pi))

            # Choose action
            action = np.random.choice(len(pi), p=pi)
            state = self.game.getNextState(state, action)

        final_len = self.game.getTourLength(state)
        examples = []
        for st, pi in trajectory:
            leftover = final_len - st.current_length
            examples.append((st, pi, leftover))

        # Check if this is the best tour found in any episode
        current_iteration = len(self.iteration_pi_loss_history) + 1  # Current iteration
        current_episode = (
            len(self.trainExamplesHistory[-1]) if self.trainExamplesHistory else 0
        )  # Current episode in this iteration

        if final_len < self.best_episode_tour_length:
            # We found a new best tour!
            log.info(
                f"New best episode tour: {final_len:.4f} (previous: {self.best_episode_tour_length:.4f})"
            )

            # Update best tour information
            self.best_episode_tour_length = final_len
            self.best_episode_tour_iteration = current_iteration
            self.best_episode_tour_episode = current_episode

            # Save the tour image
            self.save_best_episode_tour(state, current_iteration, current_episode)

        # Track best episode tour length for plotting
        if len(self.best_episode_tour_history) < current_iteration:
            self.best_episode_tour_history.append(self.best_episode_tour_length)

        # Optionally log stats about this episode
        if hasattr(self.mcts, "total_searches"):
            logging.debug(
                f"Episode unique states: {len(visited_states)}, MCTS searches: {self.mcts.total_searches}"
            )

        return examples

    def learn(self):
        # Initialize a counter for iterations with no improvement.
        self.no_improve_counter = 0

        # Initialize the champion network - use the same configuration as self.nnet
        # Create a new instance instead of deepcopy to avoid copy issues
        self.champion_nnet = type(self.nnet)(self.game, self.args)

        # Save the initial network state and load it into champion
        temp_path = os.path.join(self.args.checkpoint, "temp_initial.pth.tar")
        self.nnet.save_checkpoint(self.args.checkpoint, "temp_initial.pth.tar")
        self.champion_nnet.load_checkpoint(self.args.checkpoint, "temp_initial.pth.tar")

        # Remove the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        for i in range(1, self.args.numIters + 1):
            log.info(f"=== Starting Iter #{i} ===")
            iterationTrainExamples = []

            # Create a single MCTS instance per iteration - will be reused across episodes
            log.info(
                "Creating MCTS instance for this iteration (will be reused across episodes)"
            )
            self.mcts = MCTS(self.game, self.nnet, self.args)
            mcts_stats = {"total_nodes": 0, "cache_hits": 0}

            # Self-play
            for ep in tqdm(range(self.args.numEps), desc="Self Play"):
                # We no longer create a new MCTS for each episode
                # Track MCTS tree size before episode
                pre_episode_nodes = len(self.mcts.Ns) if hasattr(self.mcts, "Ns") else 0

                episode_data = (
                    self.executeEpisode()
                )  # list of (state, pi, leftover_dist)

                # Track MCTS tree growth
                post_episode_nodes = (
                    len(self.mcts.Ns) if hasattr(self.mcts, "Ns") else 0
                )
                nodes_added = post_episode_nodes - pre_episode_nodes
                mcts_stats["total_nodes"] = post_episode_nodes

                if hasattr(self.mcts, "cache_hits"):
                    mcts_stats["cache_hits"] = self.mcts.cache_hits

                if ep % 5 == 0:  # Log every 5 episodes
                    log.info(
                        f"Episode {ep}: MCTS tree size {post_episode_nodes} nodes (+{nodes_added} new)"
                    )

                # *** AUGMENT BEFORE ADDING TO iterationTrainExamples ***
                if getattr(self.args, "augmentationFactor", 1) > 1:
                    augmented_data = self.augmentExamples(episode_data)
                    iterationTrainExamples.extend(augmented_data)
                else:
                    iterationTrainExamples.extend(episode_data)

            # Log final MCTS statistics for this iteration
            log.info(
                f"Iteration {i} MCTS final size: {mcts_stats['total_nodes']} nodes"
            )
            if mcts_stats["cache_hits"] > 0:
                log.info(f"MCTS cache hits: {mcts_stats['cache_hits']}")

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

            # Save current champion network state as a backup
            self.champion_nnet.save_checkpoint(self.args.checkpoint, "champion.pth.tar")

            # Train the current network (self.nnet) with the new examples
            self.nnet.save_checkpoint(self.args.checkpoint, "temp.pth.tar")
            pi_loss, v_loss = self.nnet.train(trainExamples)
            self.iteration_pi_loss_history.append(pi_loss)
            self.iteration_v_loss_history.append(v_loss)

            # === COMPETITIVE EVALUATION ===
            # Evaluate both networks to compare performance
            log.info("Evaluating champion and challenger networks...")

            # Evaluate the champion network
            champion_lens = self.evaluateWithNetwork(self.champion_nnet)
            champion_avg_len = (
                float(np.mean(champion_lens)) if champion_lens else float("inf")
            )
            log.info(f"Champion avg length: {champion_avg_len:.4f}")

            # Evaluate the challenger (newly trained) network
            challenger_lens = self.evaluateAllCoords()  # Uses self.nnet
            challenger_avg_len = (
                float(np.mean(challenger_lens)) if challenger_lens else float("inf")
            )
            log.info(f"Challenger avg length: {challenger_avg_len:.4f}")

            # Store challenger results for plotting
            self.eval_avg_length_history.append(challenger_avg_len)
            self.eval_set_lengths_history.append(challenger_lens)

            # Compare the two networks
            if challenger_avg_len <= champion_avg_len:
                # Challenger is better or equal - accept the new network
                log.info(
                    f"[ACCEPTED] New network is better ({challenger_avg_len:.4f} vs {champion_avg_len:.4f})"
                )
                self.accepted_iterations.append(i)

                # Update champion network with the challenger's weights by saving and loading checkpoint
                # No need for deepcopy which may not work with some neural networks
                self.nnet.save_checkpoint(
                    self.args.checkpoint, "champion_update.pth.tar"
                )
                self.champion_nnet.load_checkpoint(
                    self.args.checkpoint, "champion_update.pth.tar"
                )

                # Clean up temporary file
                temp_path = os.path.join(
                    self.args.checkpoint, "champion_update.pth.tar"
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # If this is also better than our best so far, save as best.pth.tar
                if challenger_avg_len <= self.best_avg_length:
                    self.best_avg_length = challenger_avg_len
                    log.info("New best average => saving model as best.pth.tar")
                    self.nnet.save_checkpoint(self.args.checkpoint, "best.pth.tar")
            else:
                # Champion is better - reject the new network
                log.info(
                    f"[REJECTED] Champion network is better ({champion_avg_len:.4f} vs {challenger_avg_len:.4f})"
                )
                self.rejected_iterations.append(i)

                # Revert to champion network - load champion weights into current network
                self.champion_nnet.save_checkpoint(
                    self.args.checkpoint, "revert_to_champion.pth.tar"
                )
                self.nnet.load_checkpoint(
                    self.args.checkpoint, "revert_to_champion.pth.tar"
                )

                # Clean up temporary file
                temp_path = os.path.join(
                    self.args.checkpoint, "revert_to_champion.pth.tar"
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Update the eval history with champion's performance instead
                self.eval_avg_length_history[-1] = champion_avg_len
                self.eval_set_lengths_history[-1] = champion_lens

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
            self.plot_loss_and_length_history(i, self.eval_avg_length_history[-1])
            # Also plot acceptance/rejection history
            # self.plot_acceptance_history(i)

            # Assume you compute a variable 'current_best_tour_length' from evaluation of the iteration.
            # Lower tour length means improvement (minimization) or if you work with negative cost, then higher is better.
            # Adjust the check below based on your definitions.
            if challenger_avg_len < self.best_tour_length:
                self.best_tour_length = challenger_avg_len
                self.no_improve_counter = 0
            else:
                self.no_improve_counter += 1

            if self.no_improve_counter >= self.args.no_improvement_threshold:
                self.args.cpuct *= self.args.cpuct_update_factor
                print(
                    f"Adaptive cpuct triggered at iteration {i}: cpuct increased to {self.args.cpuct}"
                )
                self.no_improve_counter = 0

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

        # Create new distance matrix based on permutation
        if state.distance_matrix is not None:
            old_matrix = state.distance_matrix
            new_matrix = np.zeros_like(old_matrix)
            for i in range(n):
                for j in range(n):
                    new_matrix[perm[i]][perm[j]] = old_matrix[i][j]
        else:
            new_matrix = None

        new_state = TSPState(
            n,
            new_coords,
            distance_matrix=new_matrix,
            start_node=perm[state.tour[0]] if state.tour else None,
        )
        new_state.tour = new_tour
        new_state.unvisited = new_unvisited
        # if you recompute the partial cost, do:
        # new_state.current_length = self.recomputeTourLength(new_state)
        # otherwise copy:
        new_state.current_length = state.current_length

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

        # For rotation, we need to recompute the distance matrix since distances change
        # But we can also keep the same distances if simplicity is preferred
        new_state = TSPState(
            state.num_nodes,
            rotated_coords,
            distance_matrix=state.distance_matrix,  # Reuse same matrix for simplicity
            start_node=state.tour[0] if state.tour else None,
        )
        new_state.tour = list(state.tour)
        new_state.unvisited = state.unvisited.copy()
        new_state.current_length = state.current_length

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

    def evaluateWithNetwork(self, network):
        """
        Evaluate the given network on the entire coords_for_eval list.
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
            start_node = (
                self.args.get("fixed_start_node", 0)
                if self.args.get("fixed_start", True)
                else None
            )
            state = self.game.getInitState(start_node=start_node)
            temp_mcts = MCTS(self.game, network, self.args)

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
        but only if they represent an improvement over the previous best.
        Saving them in 'tours/other sets', named 'set{k}_iter_{iteration:03d}_len_{X}.png'.
        Typically called only every 'plot_all_eval_sets_interval' iteration.
        """
        log.info(f"Evaluating all stable tours for iteration {iteration} ...")

        # The 'other sets' folder
        other_folder = os.path.join(self.folder, "tours", "other sets")
        os.makedirs(other_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        for idx, coords in enumerate(self.coords_for_eval):
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

            # Check if this is an improvement (strictly better)
            if length < self.best_eval_lengths[idx]:
                log.info(
                    f"Set {idx+1}: New best length {length:.4f} (previous: {self.best_eval_lengths[idx]:.4f})"
                )
                self.best_eval_lengths[idx] = length

                # Only save plot for improvements
                filename = f"set{idx+1}_iter_{iteration:03d}_len_{length:.4f}.png"
                out_path = os.path.join(other_folder, filename)
                self.game.plotTour(
                    state,
                    title=f"EvalSet {idx+1}, Iter={iteration}, Len={length:.4f} (Improvement)",
                    save_path=out_path,
                )
            else:
                log.info(
                    f"Set {idx+1}: Length {length:.4f} is not an improvement over {self.best_eval_lengths[idx]:.4f} - skipping plot"
                )

        self.args.numMCTSSims = original_sims

    ###############################################################################
    # 1) Where you used "evaluation" folder, rename it to "tours" and store only set #1
    ###############################################################################
    def plotSingleEvalTour(self, coords, iteration, set_idx=1):
        """
        Plots the final route for coordinate set #1 each iteration,
        but only if it's an improvement over the previous best.
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

        # Check if this is an improvement (strictly better)
        set_idx_zero_based = set_idx - 1  # Convert 1-based to 0-based index
        if length < self.best_eval_lengths[set_idx_zero_based]:
            log.info(
                f"Set {set_idx}: New best length {length:.4f} (previous: {self.best_eval_lengths[set_idx_zero_based]:.4f})"
            )
            self.best_eval_lengths[set_idx_zero_based] = length

            # Only save plot for improvements
            filename = f"iter_{iteration:03d}_len_{length:.4f}.png"
            out_path = os.path.join(tours_folder, filename)

            self.game.plotTour(
                state,
                title=f"Set{set_idx}, Iter={iteration}, Len={length:.4f} (Improvement)",
                save_path=out_path,
            )
        else:
            log.info(
                f"Set {set_idx}: Length {length:.4f} is not an improvement over {self.best_eval_lengths[set_idx_zero_based]:.4f} - skipping plot"
            )

        self.args.numMCTSSims = original_sims

    def plot_acceptance_history(self, iteration):
        """
        Create a plot showing which iterations were accepted and rejected.
        """
        fig, ax = plt.subplots(figsize=(12, 3))

        # Plot all iterations
        all_iters = list(range(1, iteration + 1))
        ax.scatter(
            all_iters,
            [0] * len(all_iters),
            color="grey",
            alpha=0.3,
            s=50,
            label="All Iterations",
        )

        # Plot accepted iterations
        if self.accepted_iterations:
            ax.scatter(
                self.accepted_iterations,
                [0] * len(self.accepted_iterations),
                color="green",
                marker="^",
                s=100,
                label="Accepted",
            )

        # Plot rejected iterations
        if self.rejected_iterations:
            ax.scatter(
                self.rejected_iterations,
                [0] * len(self.rejected_iterations),
                color="red",
                marker="x",
                s=100,
                label="Rejected",
            )

        # Remove y-axis ticks since this is just a timeline
        ax.set_yticks([])
        ax.set_xlabel("Iteration")
        ax.set_title("Network Acceptance/Rejection History")
        ax.grid(axis="x", alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

        # Save plot
        acceptance_plot_path = os.path.join(self.folder, f"acceptance_history.png")
        fig.savefig(acceptance_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

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
            label="Evaluation Length",
            color="blue",
            linewidth=2,
        )

        # Mark accepted and (NOT rejected iterations)
        for i in self.accepted_iterations:
            if i <= len(self.eval_avg_length_history):
                axL.scatter(
                    i,
                    self.eval_avg_length_history[i - 1],
                    color="blue",
                    marker="^",
                    s=80,
                )

        # Add best episode tour length as a separate line if we have history
        if self.best_episode_tour_history:
            # Make sure the history matches the number of iterations
            while len(self.best_episode_tour_history) < len(iters):
                self.best_episode_tour_history.append(
                    self.best_episode_tour_history[-1]
                )

            axL.plot(
                iters,
                self.best_episode_tour_history,
                label="Best Episode Tour",
                color="green",
                linewidth=1.5,
                linestyle="--",
            )

            # Add a marker for the current best episode tour
            if self.best_episode_tour_iteration is not None:
                axL.scatter(
                    self.best_episode_tour_iteration,
                    self.best_episode_tour_length,
                    color="green",
                    marker="*",
                    s=120,
                    zorder=10,
                    label=f"Best Episode ({self.best_episode_tour_length:.1f})",
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

        # Update the title to include both eval and best episode lengths
        if self.best_tour_length is not None:
            eval_gap = (
                (eval_len - self.best_tour_length) / self.best_tour_length
            ) * 100
            ep_gap = (
                (self.best_episode_tour_length - self.best_tour_length)
                / self.best_tour_length
            ) * 100
            title = f"Iter {iteration}: Eval={eval_len:.1f} (Gap: {eval_gap:.1f}%), Best Episode={self.best_episode_tour_length:.1f} (Gap: {ep_gap:.1f}%)"
        else:
            title = f"Iter {iteration}: Eval={eval_len:.1f}, Best Episode={self.best_episode_tour_length:.1f}"

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

    def save_best_episode_tour(self, state, iteration, episode):
        """
        Save the best tour found during any episode to the tours folder.
        """
        tours_folder = os.path.join(self.folder, "tours")
        os.makedirs(tours_folder, exist_ok=True)

        # Format the filename according to the requested pattern
        filename = f"iter_{iteration:03d}_episode_{episode}_len_{self.best_episode_tour_length:.4f}.png"
        out_path = os.path.join(tours_folder, filename)

        # Plot and save the tour
        self.game.plotTour(
            state,
            title=f"Best Tour (Iter={iteration}, Ep={episode}, Len={self.best_episode_tour_length:.4f})",
            save_path=out_path,
        )

        log.info(f"Saved best episode tour image: {filename}")
