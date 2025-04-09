import logging
import os
import csv
import copy
from pickle import Pickler, Unpickler
from random import shuffle
import concurrent.futures

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

        # Store the initial cpuct value from args
        self.initial_cpuct = self.args.cpuct
        # Define the cpuct to use during evaluation (default to initial, or allow override via args)
        self.eval_cpuct = self.args.get("eval_cpuct", self.initial_cpuct)
        log.info(
            f"Initial cpuct: {self.initial_cpuct}, Evaluation cpuct: {self.eval_cpuct}"
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
            # Evaluate from configured start node
            start_node = (
                self.args.get("fixed_start_node", 0)
                if self.args.get("fixed_start", True)
                else np.random.choice(len(coords))
            )
            self.game.node_coordinates = coords
            state = self.game.getInitState(start_node=start_node)
            temp_mcts = MCTS(
                self.game, self.nnet, self.args, cpuct_override=self.eval_cpuct
            )

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
            all_train_examples = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.numEpisodesParallel
            ) as executor:
                # Submit self-play episodes in parallel.
                futures = [
                    executor.submit(self.executeEpisode)
                    for _ in range(self.args.numEps)
                ]
                for future in concurrent.futures.as_completed(futures):
                    episode_examples = future.result()
                    all_train_examples.extend(episode_examples)

            # Track MCTS tree growth
            post_episode_nodes = len(self.mcts.Ns) if hasattr(self.mcts, "Ns") else 0
            nodes_added = post_episode_nodes - len(self.mcts.Ns)
            mcts_stats["total_nodes"] = post_episode_nodes

            if hasattr(self.mcts, "cache_hits"):
                mcts_stats["cache_hits"] = self.mcts.cache_hits

            if len(self.mcts.Ns) > 0:
                log.info(f"Iteration {i} MCTS final size: {len(self.mcts.Ns)} nodes")
            if mcts_stats["cache_hits"] > 0:
                log.info(f"MCTS cache hits: {mcts_stats['cache_hits']}")

            # Now add the iterationTrainExamples to trainExamplesHistory
            self.trainExamplesHistory.append(all_train_examples)

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

            # Store challenger results for plotting before potential reversion
            self.eval_avg_length_history.append(challenger_avg_len)
            self.eval_set_lengths_history.append(
                challenger_lens
            )  # Store challenger's detailed lengths

            # Compare the two networks
            network_accepted = False  # Flag to track acceptance
            if challenger_avg_len <= champion_avg_len:
                # Challenger wins the head-to-head comparison for this iteration
                log.info(
                    f"[ACCEPTED Head-to-Head] New network better/equal ({challenger_avg_len:.4f} vs {champion_avg_len:.4f})"
                )
                network_accepted = True
                self.accepted_iterations.append(i)

                # --- Reset cpuct on acceptance ---
                if self.args.cpuct != self.initial_cpuct:
                    log.info(
                        f"Resetting adaptive cpuct from {self.args.cpuct} back to initial {self.initial_cpuct}"
                    )
                    self.args.cpuct = self.initial_cpuct
                # ---------------------------------

                # **CRITICAL CHANGE:** Only update historical best and save 'best.pth.tar'
                # if the challenger is better than the *overall best recorded average*
                if challenger_avg_len < self.best_avg_length:
                    log.info(
                        f"*** New Overall Best Average: {challenger_avg_len:.4f} (previous: {self.best_avg_length:.4f}) ***"
                    )
                    self.best_avg_length = challenger_avg_len
                    log.info("Saving new best model as best.pth.tar")
                    # Save the current network (challenger) as the best
                self.nnet.save_checkpoint(self.args.checkpoint, "best.pth.tar")
                # else: # Challenger won head-to-head but wasn't a new overall best
                # log.info(f"Challenger accepted but not a new overall best ({challenger_avg_len:.4f} vs best {self.best_avg_length:.4f})")
                # pass # No need to save best.pth.tar

                # Update the champion network to become the challenger for the *next* iteration's comparison
                # No need for deepcopy which may not work with some neural networks
                self.nnet.save_checkpoint(
                    self.args.checkpoint,
                    "champion_update.pth.tar",  # Save current nnet state
                )
                self.champion_nnet.load_checkpoint(
                    self.args.checkpoint,
                    "champion_update.pth.tar",  # Load it into champion
                )
                # Clean up temporary file
                temp_path = os.path.join(
                    self.args.checkpoint, "champion_update.pth.tar"
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Reset improvement counter since the network weights changed
                self.no_improve_counter = 0

            else:
                # Champion wins the head-to-head comparison
                log.info(
                    f"[REJECTED Head-to-Head] Champion network is better ({champion_avg_len:.4f} vs {challenger_avg_len:.4f})"
                )
                network_accepted = False
                self.rejected_iterations.append(i)

                # Revert current network (self.nnet) back to the champion's weights
                self.champion_nnet.save_checkpoint(
                    self.args.checkpoint,
                    "revert_to_champion.pth.tar",  # Save champion state
                )
                self.nnet.load_checkpoint(
                    self.args.checkpoint,
                    "revert_to_champion.pth.tar",  # Load it back into nnet
                )
                # Clean up temporary file
                temp_path = os.path.join(
                    self.args.checkpoint, "revert_to_champion.pth.tar"
                )
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # Log the champion's performance again for this iteration's history
                # (since the challenger was rejected, the performance for this iter is the champion's)
                self.eval_avg_length_history[-1] = champion_avg_len
                self.eval_set_lengths_history[-1] = (
                    champion_lens  # Overwrite with champion's detailed lengths
                )

                # Increment no improvement counter only if challenger was rejected
                self.no_improve_counter += 1

            # --- Adaptive CPUCT (moved after acceptance/rejection logic) ---
            if (
                not network_accepted
                and self.no_improve_counter >= self.args.no_improvement_threshold
            ):
                old_cpuct = self.args.cpuct
                self.args.cpuct *= self.args.cpuct_update_factor
                log.info(
                    f"Adaptive cpuct triggered (no improvement for {self.no_improve_counter} iters): cpuct increased from {old_cpuct:.4f} to {self.args.cpuct:.4f}"
                )
                self.no_improve_counter = 0  # Reset after triggering

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
            # if challenger_avg_len < self.best_tour_length:
            #     self.best_tour_length = challenger_avg_len
            #     self.no_improve_counter = 0
            # else:
            #     self.no_improve_counter += 1


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
            temp_mcts = MCTS(
                self.game, self.nnet, self.args, cpuct_override=self.eval_cpuct
            )

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
            temp_mcts = MCTS(
                self.game, network, self.args, cpuct_override=self.eval_cpuct
            )

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
        but only if they represent an improvement over the previous best. <-- Now enforces saving only on improvement
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
            # Ensure fixed start node logic is consistent if needed
            start_node = (
                self.args.get("fixed_start_node", 0)
                if self.args.get("fixed_start", True)
                else None
            )
            state = self.game.getInitState(start_node=start_node)
            temp_mcts = MCTS(
                self.game, self.nnet, self.args, cpuct_override=self.eval_cpuct
            )

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
                    f"Set {idx+1}: New best length {length:.4f} (previous: {self.best_eval_lengths[idx]:.4f}) - SAVING PLOT"
                )
                self.best_eval_lengths[idx] = length

                # --- Only save plot for improvements ---
                filename = f"set{idx+1}_iter_{iteration:03d}_len_{length:.4f}.png"
                out_path = os.path.join(other_folder, filename)
                self.game.plotTour(
                    state,
                    title=f"EvalSet {idx+1}, Iter={iteration}, Len={length:.4f} (Improvement)",
                    save_path=out_path,
                )
                # ----------------------------------------
            else:
                log.info(
                    f"Set {idx+1}: Length {length:.4f} is not an improvement over {self.best_eval_lengths[idx]:.4f} - skipping plot"
                )
                # --- Remove plotting logic from here ---
                # filename = f"set{idx+1}_iter_{iteration:03d}_len_{length:.4f}.png" # No longer needed
                # out_path = os.path.join(other_folder, filename) # No longer needed
                # self.game.plotTour(...) # Remove this call

        self.args.numMCTSSims = original_sims

    ###############################################################################
    # 1) Where you used "evaluation" folder, rename it to "tours" and store only set #1
    ###############################################################################
    def plotSingleEvalTour(self, coords, iteration, set_idx=1):
        """
        Plots the final route for coordinate set #1 each iteration,
        but only if it's an improvement over the previous best.  <-- Now enforces saving only on improvement
        Output file in 'tours/' => 'iter_{iteration:03d}_len_{length:.4f}.png'
        """
        tours_folder = os.path.join(self.folder, "tours")
        os.makedirs(tours_folder, exist_ok=True)

        original_sims = self.args.numMCTSSims
        self.args.numMCTSSims = self.args.numMCTSSimsEval

        self.game.node_coordinates = coords
        # Ensure fixed start node logic is consistent if needed
        start_node = (
            self.args.get("fixed_start_node", 0)
            if self.args.get("fixed_start", True)
            else None
        )
        state = self.game.getInitState(start_node=start_node)
        temp_mcts = MCTS(
            self.game, self.nnet, self.args, cpuct_override=self.eval_cpuct
        )

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
                f"Set {set_idx}: New best length {length:.4f} (previous: {self.best_eval_lengths[set_idx_zero_based]:.4f}) - SAVING PLOT"
            )
            self.best_eval_lengths[set_idx_zero_based] = length

            # --- Only save plot for improvements ---
            filename = f"iter_{iteration:03d}_len_{length:.4f}.png"
            out_path = os.path.join(tours_folder, filename)

            self.game.plotTour(
                state,
                title=f"Set{set_idx}, Iter={iteration}, Len={length:.4f} (Improvement)",
                save_path=out_path,
            )
            # ----------------------------------------
        else:
            log.info(
                f"Set {set_idx}: Length {length:.4f} is not an improvement over {self.best_eval_lengths[set_idx_zero_based]:.4f} - skipping plot"
            )
            # --- Remove plotting logic from here ---
            # filename = f"iter_{iteration:03d}_len_{length:.4f}.png" # No longer needed here
            # out_path = os.path.join(tours_folder, filename) # No longer needed here
            # self.game.plotTour(...) # Remove this call
            # log.info(...) # Remove this log message about saving plot

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
