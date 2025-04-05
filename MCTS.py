import logging
import math
import numpy as np
import TSPState
import TSPGame
import NNetWrapper
from tqdm import tqdm

EPS = 1e-8
log = logging.getLogger(__name__)


class MCTS:
    """
    MCTS for single-player TSP:
    - No visited sets.
    - No player parameter.
    - getGameEnded returns final outcome if terminal, else 0.
    """

    def __init__(self, game: TSPGame, nnet: NNetWrapper, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.Pred_cache = {}  # Cache for predictions

        # Statistics for monitoring
        self.cache_hits = 0
        self.total_searches = 0
        self.tree_depth = 0

    def getActionProb(self, state, temp=1):
        if self.args.explicit_prints:
            log.info("=== Getting action probabilities ===")
            log.info(f"Temperature: {temp}")

        for i in range(self.args.numMCTSSims):
            if self.args.explicit_prints:
                log.info(f"Starting simulation {i+1}/{self.args.numMCTSSims}")
            self.search(state)

        s = self.game.uniqueStringRepresentation(state)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

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

        if state_string not in self.Es:
            if self.game.isTerminal(tsp_state):  # Terminal
                self.Es[state_string] = self.game.getFinalScore(tsp_state)
                if self.args.explicit_prints:
                    log.info(f"Game ended with value: {self.Es[state_string]}")
                return self.Es[state_string]
            else:
                self.Es[state_string] = None  # Not terminal
        if self.Es[state_string] != None:
            # terminal node
            if self.args.explicit_prints:
                log.info(f"Game ended with value: {self.Es[state_string]}")
            return self.Es[state_string]

        if state_string not in self.Ps:
            # New leaf node
            if self.args.explicit_prints:
                log.info("New state encountered - getting policy from neural network")
            if state_string in self.Pred_cache:
                self.Ps[state_string], leftover_v = self.Pred_cache[state_string]
                self.cache_hits += 1
            else:
                self.Ps[state_string], leftover_v = self.nnet.predict(tsp_state)
                self.Pred_cache[state_string] = (self.Ps[state_string], leftover_v)

            total_cost = tsp_state.current_length + leftover_v
            v = -total_cost

            valids = self.game.getValidMoves(tsp_state)
            self.Ps[state_string] = self.Ps[state_string] * valids
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
            if self.args.explicit_prints:
                log.info(f"NN Predicted Remaining Cost: {leftover_v:.3f}")
                log.info(f"Estimated Total Cost (v): {(total_cost):.3f}")
                log.info(
                    "Policy probabilities:",
                    {
                        i: f"{p:.3f}"
                        for i, p in enumerate(self.Ps[state_string])
                        if p > 0
                    },
                )
            return v

        # Internal Node
        valids = self.Vs[state_string]
        cur_best = -float("inf")
        best_act = -1

        if self.args.explicit_prints:
            log.info(
                f"Evaluating moves for state {state_string} with Ns(s): {self.Ns[state_string]} visits:"
            )

        for a in range(self.game.getActionSize()):
            if valids[a]:
                discount = 1
                q_value = self.Qsa.get((state_string, a), 0)
                if (state_string, a) in self.Qsa:
                    discount = (-0.01
                        * (
                            self.args.cpuct
                            * self.Ns[state_string]
                            ** (1 / (1 + self.Nsa[(state_string, a)]))
                        )+ 1)
                    u = q_value * (
                        discount
                        # - self.Ps[state_string][a]
                    )

                    # u = q_value + (
                    #     self.args.cpuct
                    #     * self.Ps[state_string][a]
                    #     * math.sqrt(self.Ns[state_string])
                    #     / (1 + self.Nsa[(state_string, a)])
                    # )
                else:
                    u = (
                        self.args.cpuct
                        * self.Ps[state_string][a]
                        * math.sqrt(self.Ns[state_string] + EPS)
                    )

                if self.args.explicit_prints:
                    log.info(
                        f" Action {a}: Q-value: {q_value:.3f}, Nsa(s,a): {self.Nsa.get((state_string, a), 0)}, Prior P: {self.Ps[state_string][a]:.3f}, discount: {discount:.3f}, Selection Score (u): {u:.3f}"
                    )

                if u > cur_best:
                    cur_best = u
                    best_act = a
                    if self.args.explicit_prints:
                        log.info(f"  -> New best action!")

        a = best_act
        if self.args.explicit_prints:
            log.info(f"Selected action: {a} with PUCT value: {cur_best:.3f}")

        next_s = self.game.getNextState(tsp_state, a)
        v = self.search(next_s)

        if (state_string, a) in self.Qsa:
            # Average the new value v with the existing Q-value                    log.info("All valid moves were masked, defaulting to uniform policy")
            # self.Qsa[(state_string, a)] = max(self.Qsa[(state_string, a)], v)
            
            # if we find a better value, we update the Q-value
            # if we find a worse value, we save the avarge
            if(self.Qsa[(state_string, a)]<v):
                self.Qsa[(state_string, a)] = v
            else:
                self.Qsa[(state_string, a)] = (
                    self.Nsa[(state_string, a)] * self.Qsa[(state_string, a)] + v
                ) / (self.Nsa[(state_string, a)] + 1)
            self.Nsa[(state_string, a)] += 1
        else:
            self.Qsa[(state_string, a)] = v
            self.Nsa[(state_string, a)] = 1

        self.Ns[state_string] += 1
        if self.args.explicit_prints:
            log.info(f"Backpropagating value: {v}")
            log.info(
                f"Updated Q-value for (s:{state_string}, a:{a}): {self.Qsa[(state_string, a)]:.3f}"
            )
            log.info(
                f"Updated visit count for (s:{state_string}, a:{a}): {self.Nsa[(state_string, a)]}"
            )
            log.info(f"Updated visit count for state: {self.Ns[state_string]}")
        return v
