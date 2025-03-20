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
        uniqueString = self.game.uniqueStringRepresentation(state)

        # Run simulations to build the search tree
        for _ in range(self.args.numMCTSSims):
            self.search(state, uniqueString)  # Pass the precomputed string
            self.total_searches += 1

        counts = [
            self.Nsa[(uniqueString, a)] if (uniqueString, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
        else:
            counts_exp = [x ** (1.0 / temp) for x in counts]
            counts_sum = float(sum(counts_exp))
            probs = [x / counts_sum for x in counts_exp]

        return probs

    def search(self, tsp_state: TSPState, state_str=None):
        state_string = (
            state_str
            if state_str is not None
            else self.game.uniqueStringRepresentation(tsp_state)
        )

        if state_string not in self.Es:
            if self.game.isTerminal(tsp_state):  # Terminal
                self.Es[state_string] = self.game.getFinalScore(tsp_state)
                return self.Es[state_string]
            else:
                self.Es[state_string] = None  # Not terminal
        if self.Es[state_string] != None:
            # terminal node
            return self.Es[state_string]

        if state_string not in self.Ps:
            # Leaf node
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
                print(self.Ps[state_string])
                log.error("All valid moves were masked, assigning equal probabilities.")
                self.Ps[state_string] = self.Ps[state_string] + valids
                self.Ps[state_string] /= np.sum(self.Ps[state_string])

            self.Vs[state_string] = valids
            self.Ns[state_string] = 0
            return v

        # Internal Node
        valids = self.Vs[state_string]
        cur_best = -float("inf")
        best_act = -1

        # UCB
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (state_string, a) in self.Qsa:
                    u = self.Qsa[(state_string, a)] + self.args.cpuct * self.Ps[
                        state_string
                    ][a] * math.sqrt(self.Ns[state_string]) / (
                        1 + self.Nsa[(state_string, a)]
                    )
                else:
                    u = (
                        self.args.cpuct
                        * self.Ps[state_string][a]
                        * math.sqrt(self.Ns[state_string] + EPS)
                    )
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.getNextState(tsp_state, a)
        # Do best expected action
        v = self.search(next_s)

        # Update Q, N - Avarage of old Qsa and new v
        if (state_string, a) in self.Qsa:
            # Keep maximum value for deterministic environments
            self.Qsa[(state_string, a)] = max(self.Qsa[(state_string, a)], v)
            self.Nsa[(state_string, a)] += 1
        else:
            self.Qsa[(state_string, a)] = v
            self.Nsa[(state_string, a)] = 1

        self.Ns[state_string] += 1
        return v

    def simulate(self, state):
        # ... existing simulation code ...
        if random.random() < 0.2:  # 20% chance to apply local search
            improved_state = self.two_opt_improvement(state)
            return improved_state.current_length

    def two_opt_improvement(self, state):
        best = state.current_length
        improved = True
        while improved:
            improved = False
            for i in range(len(state.tour) - 1):
                for j in range(i + 2, len(state.tour)):
                    # Try 2-opt swap
                    new_length = self.calculate_swap(state, i, j)
                    if new_length < best:
                        best = new_length
                        improved = True
        return state
