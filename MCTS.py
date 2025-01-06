import logging
import math
import numpy as np
from TSP import TSPState
from TSP.TSPGame import TSPGame
from TSP.pytorch.NNetWrapper import NNetWrapper

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

    def getActionProb(self, canonicalBoard, temp=1):
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.uniqueStringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]
        # print("counts",counts)

        if temp == 0:
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        else:
            counts_exp = [x ** (1.0 / temp) for x in counts]
            counts_sum = float(sum(counts_exp))
            probs = [x / counts_sum for x in counts_exp]
            return probs

    def search(self, tsp_state: TSPState):
        s = self.game.uniqueStringRepresentation(tsp_state)
        # Dirichlet Noise for Better Exploration: AlphaZero adds Dirichlet noise
        # to Ps[s] at the root node to ensure initial exploration of multiple moves rather
        # than getting stuck too early. This code doesn't seem to add such noise. Adding:

        # python
        # Copy code
        # if is_root_node:
        #     Ps[s] = (1 - noise_alpha)*Ps[s] + noise_alpha*dirichlet_noise

        if s not in self.Es:
            if self.game.isTerminal(tsp_state): # Terminal
                self.Es[s] = self.game.getFinalScore(tsp_state)
                return self.Es[s]
            else:
                self.Es[s] = None # Not terminal
        if self.Es[s] != None:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # Leaf node
            self.Ps[s], v = self.nnet.predict(tsp_state)
            valids = self.game.getValidMoves(tsp_state)

            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                log.error("All valid moves were masked, assigning equal probabilities.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        # Internal Node
        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # UCB
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = self.game.getNextState(tsp_state, a)
        # Do best expected action
        v = self.search(next_s)

        # Update Q, N - Avarage of old Qsa and new v
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
