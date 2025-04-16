import logging
import math
import numpy as np
import TSPState
import TSPGame
import NNetWrapper
from tqdm import tqdm
from utils import complete_tour_with_lin_kernighan, complete_tour_with_nearest_neighbor
import threading

EPS = 1e-8
log = logging.getLogger(__name__)


class MCTS:
    """
    MCTS for single-player TSP:
    - No visited sets.
    - No player parameter.
    - getGameEnded returns final outcome if terminal, else 0.
    """

    def __init__(self, game: TSPGame, nnet: NNetWrapper, args, cpuct_override=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        # Use override if provided, otherwise use the (potentially adaptive) value from args
        self.cpuct = cpuct_override if cpuct_override is not None else self.args.cpuct
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.state_value_prediction = {}
        self.Pred_cache = {}  # Cache for predictions

        # Statistics for monitoring
        self.cache_hits = 0
        self.total_searches = 0
        self.tree_depth = 0

        # Add Lock
        self.lock = threading.Lock()

    def getActionProb(self, state, temp=1):
        if self.args.explicit_prints:
            log.info("=== Getting action probabilities ===")
            log.info(f"Temperature: {temp}")

        for i in range(self.args.numMCTSSims):
            if self.args.explicit_prints:
                log.info(f"Starting simulation {i+1}/{self.args.numMCTSSims}")
            self.search(state)

        s = self.game.uniqueStringRepresentation(state)
        counts = [0] * self.game.getActionSize()

        # --- Read Nsa (Locked Read) ---
        with self.lock:
            for a in range(self.game.getActionSize()):
                counts[a] = self.Nsa.get((s, a), 0)  # Read counts inside lock
        # --- End Read ---

        if self.args.explicit_prints:
            log.info("Visit counts for each action:")
            for a, count in enumerate(counts):
                if count > 0:
                    log.info(f"Action {a}: {count} visits")

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            if self.args.explicit_prints:
                log.info(f"Temp=0: Selecting best action {bestA}")
        else:
            counts = [x ** (1.0 / temp) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]
            if self.args.explicit_prints:
                log.info("Action probabilities:")
                for a, prob in enumerate(probs):
                    if prob > 0:
                        log.info(f"Action {a}: {prob:.3f}")

        return probs

    def search(self, tsp_state: TSPState, state_str=None):
        if self.args.explicit_prints:
            log.info("=== Starting new MCTS search ===")
            log.info(
                f"Current board state: {self.game.uniqueStringRepresentation(tsp_state)}"
            )

        state_string = (
            state_str
            if state_str is not None
            else self.game.uniqueStringRepresentation(tsp_state)
        )

        # --- Check Terminal State (Locked Read/Write) ---
        with self.lock:
            terminal_value = self.Es.get(
                state_string
            )  # Use .get for safer access initially

        if terminal_value is not None:  # Already known terminal state
            if self.args.explicit_prints:
                log.info(f"Game ended with value: {terminal_value}")
            return terminal_value

        is_terminal = self.game.isTerminal(tsp_state)
        if is_terminal:
            final_score = self.game.getFinalScore(tsp_state)
            with self.lock:
                self.Es[state_string] = final_score  # Store terminal state value
            if self.args.explicit_prints:
                log.info(f"Game ended with value: {final_score}")
            return final_score
        # --- End Terminal State Check ---

        # --- Check if Leaf Node (Locked Read) ---
        with self.lock:
            policy_exists = state_string in self.Ps
            # If it exists, read necessary values for internal node path safely
            if policy_exists:
                current_valids = self.Vs.get(state_string)  # Use .get for safety
                current_Ns = self.Ns.get(state_string, 0)  # Default to 0
                current_state_value_pred = self.state_value_prediction.get(state_string)
            else:
                # Check cache inside lock
                cached_pred = self.Pred_cache.get(state_string)

        # --- Handle New Leaf Node ---
        if not policy_exists:
            if self.args.explicit_prints:
                log.info("New state encountered - getting policy from neural network")
            if cached_pred:
                policy, leftover_v = cached_pred
                with self.lock:  # Update cache hits under lock
                    self.cache_hits += 1
            else:
                # *** Predict OUTSIDE the lock ***
                policy, leftover_v = self.nnet.predict(tsp_state)
                # *** Re-acquire lock to store prediction ***
                with self.lock:
                    self.Pred_cache[state_string] = (policy, leftover_v)

            total_cost = tsp_state.current_length + leftover_v
            v = -total_cost
            valids = self.game.getValidMoves(
                tsp_state
            )  # getValidMoves should be thread-safe if it only reads game state

            # --- Update Shared State (Locked Write) ---
            with self.lock:
                # Re-check if another thread added it while we were predicting
                if state_string not in self.Ps:
                    self.Ps[state_string] = policy * valids  # Apply valids mask
                    sum_Ps_s = np.sum(self.Ps[state_string])
                    if sum_Ps_s > 0:
                        self.Ps[state_string] /= sum_Ps_s
                    else:
                        if self.args.explicit_prints:
                            log.info(
                                "All valid moves were masked, defaulting to uniform policy"
                            )
                        self.Ps[state_string] = self.Ps[state_string] + valids
                        self.Ps[state_string] /= np.sum(self.Ps[state_string])

                    self.Vs[state_string] = valids
                    self.Ns[state_string] = 0
                    self.state_value_prediction[state_string] = v
                else:
                    # Another thread beat us, we need to proceed as if it was an internal node
                    # Read the values set by the other thread
                    current_valids = self.Vs.get(state_string)
                    current_Ns = self.Ns.get(state_string, 0)
                    current_state_value_pred = self.state_value_prediction.get(
                        state_string
                    )
                    policy_exists = True  # Update flag

            # ... (logging with potentially updated v) ...
            # If we still just created the node, return its value
            if not policy_exists:  # Check flag again
                return v
            # Otherwise fall through to internal node logic below
        # --- End New Leaf Node ---

        # --- Internal Node Logic ---
        # Use values read safely under lock earlier (current_valids, current_Ns, etc.)
        if current_valids is None:
            # This indicates a potential logic issue if policy_exists but valids is None
            log.error(
                f"Race condition recovery issue or logic error: Policy exists for {state_string} but Vs is missing."
            )
            # Handle error appropriately, maybe return a default value or raise exception
            # For now, let's try to recover by getting valids again (though ideally this shouldn't happen)
            current_valids = self.game.getValidMoves(tsp_state)

        cur_best = -float("inf")
        best_act = -1

        # --- Action Selection Loop (Locked Reads) ---
        with self.lock:
            # ... (logging) ...
            if current_state_value_pred is None:
                # Handle potential recovery issue if state_value_prediction is missing
                log.error(
                    f"Race condition recovery issue or logic error: state_value_prediction missing for {state_string}"
                )
                # Use a default or re-predict (less ideal)
                _, svp_temp = self.nnet.predict(tsp_state)  # Predict again as fallback
                current_state_value_pred = -(tsp_state.current_length + svp_temp)

            for a in range(self.game.getActionSize()):
                if current_valids[a]:
                    q_value = self.Qsa.get(
                        (state_string, a), current_state_value_pred
                    )  # Use state value pred if Q(s,a) unknown
                    nsa = self.Nsa.get((state_string, a), 0)
                    prior_p = self.Ps[state_string][a]  # Ps should exist if we are here

                    # --- Your PUCT logic ---
                    # Using Ns and Nsa read inside the lock
                    discount = (
                        -0.01 * (self.args.cpuct * current_Ns ** (1 / (1 + nsa)))
                        + 1
                        - prior_p
                    )  # Use current_Ns
                    u = q_value * discount
                    # -----------------------

                    # ... (logging uct value) ...

                    if u > cur_best:
                        cur_best = u
                        best_act = a
                        # ... (logging new best) ...
        # --- End Action Selection Loop ---

        # --- If no valid moves found (shouldn't happen unless terminal?) ---
        if best_act == -1:
            log.warning(
                f"No valid action found for non-terminal state {state_string}. Valid moves: {current_valids}"
            )
            # Handle this - maybe return 0 or state value prediction?
            return (
                current_state_value_pred if current_state_value_pred is not None else 0
            )

        a = best_act
        # ... (logging selected action) ...

        next_s = self.game.getNextState(
            tsp_state, a
        )  # getNextState should be thread-safe

        # *** Recursive call OUTSIDE the lock ***
        v = self.search(next_s)

        # --- Update Qsa, Nsa, Ns (Locked Write) ---
        with self.lock:
            if (state_string, a) in self.Qsa:
                # Your update logic (averaging or taking max)
                self.Qsa[(state_string, a)] = max(self.Qsa[(state_string, a)], v)
                # self.Qsa[(state_string, a)] = (self.Nsa[(state_string, a)] * self.Qsa[(state_string, a)] + v) / (self.Nsa[(state_string, a)] + 1)
                self.Nsa[(state_string, a)] += 1
            else:
                self.Qsa[(state_string, a)] = v
                self.Nsa[(state_string, a)] = 1

            self.Ns[state_string] = self.Ns.get(state_string, 0) + 1  # Update Ns safely
        # --- End Update ---

        # ... (logging backpropagating value) ...
        return v
