import logging
import math
import numpy as np
from TSP.TSPGame import TSPGame
from TSP.TSPState import TSPState
from TSP.pytorch.NNetWrapper import NNetWrapper

EPS = 1e-8

log = logging.getLogger(__name__)



class MCTS:
    """
    This class handles the MCTS tree for the Traveling Salesman Problem (TSP).
    """

    def __init__(self, game: TSPGame, nnet: NNetWrapper, args):
        self.game = game
        self.nnetWrapper = nnet
        self.args = args
        self.Q_state_action = {}  # Stores Q values for state-action pairs
        self.Visits_state_action = {}  # Stores visit counts for state-action pairs
        self.Visits_state = {}  # Stores visit counts for states
        self.Policy_state = {}  # Stores initial policy (returned by neural net)

        self.Valid_moves_state = {}  # Stores valid moves for states

    def getActionProb(self, tsp_state, temp=1):
        """
        Performs MCTS simulations and returns the action probabilities.

        Args:
            canonicalBoard (np.array): The current state in canonical form.
            temp (float): Temperature parameter for exploration.

        Returns:
            probs (list): A list of action probabilities.
        """
        for _ in range(self.args.numMCTSSims):
            self.search(tsp_state)

        stateKey = self.game.uniqueStringRepresentation(tsp_state)
        counts = [
            self.Visits_state_action.get((stateKey, a), 0)
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts_exp = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts_exp))
        probs = [x / counts_sum for x in counts_exp]
        return probs

    def search(self, tsp_state: TSPState, depth=0, visited=None):
        if visited is None:
            visited = set()

        stateKey = self.game.uniqueStringRepresentation(tsp_state)

        if stateKey in visited:
            # Cycle detected
            # print("Cycle detected at state:", stateKey)
            _, v = self.nnetWrapper.predict(tsp_state)
            visited.discard(stateKey)
            return v

        visited.add(stateKey)

        MAX_DEPTH = self.args.maxDepth
        if depth >= MAX_DEPTH:
            # Depth limit reached
            _, v = self.nnetWrapper.predict(tsp_state)
            visited.discard(stateKey)
            return v

        if stateKey not in self.Policy_state:
            # Leaf node: expand and evaluate
            self.Policy_state[stateKey], v = self.nnetWrapper.predict(tsp_state)
            valids = self.game.getValidMoves(tsp_state)
            self.Policy_state[stateKey] = self.Policy_state[stateKey] * valids  # Mask invalid moves
            sum_Ps_s = np.sum(self.Policy_state[stateKey])
            if sum_Ps_s > 0:
                self.Policy_state[stateKey] /= sum_Ps_s  # Renormalize
            else:
                # Handle case where all moves are invalid
                log.error("All valid moves were masked, assigning equal probabilities.")
                self.Policy_state[stateKey] = valids / np.sum(valids)

            self.Valid_moves_state[stateKey] = valids
            self.Visits_state[stateKey] = 0
            visited.discard(stateKey)
            return v  # Return the value from the leaf node

        valids = self.Valid_moves_state[stateKey]
        cur_best = -float("inf")
        best_act = -1

        # Select the action with the highest UCB value
        for action in range(self.game.getActionSize()):
            if valids[action]:
                if (stateKey, action) in self.Q_state_action:
                    u = self.Q_state_action[
                        (stateKey, action)
                    ] + self.args.cpuct * self.Policy_state[stateKey][
                        action
                    ] * math.sqrt(
                        self.Visits_state[stateKey]
                    ) / (
                        1 + self.Visits_state_action[(stateKey, action)]
                    )
                else:
                    u = (
                        self.args.cpuct
                        * self.Policy_state[stateKey][action]
                        * math.sqrt(self.Visits_state[stateKey] + EPS)
                    )
                if u > cur_best:
                    cur_best = u
                    best_act = action

        if best_act == -1:
            # No valid action found
            print("No valid action found at state:", stateKey)
            visited.discard(stateKey)
            return 0  # Or an appropriate heuristic value

        action = best_act
        next_s = self.game.getNextState(tsp_state, action)
        v = self.search(next_s, depth + 1, visited)

        # Update Qsa, Nsa values
        sa = (stateKey, action)
        if sa in self.Q_state_action:
            self.Q_state_action[sa] = (self.Visits_state_action[sa] * self.Q_state_action[sa] + v) / (self.Visits_state_action[sa] + 1)
            self.Visits_state_action[sa] += 1
        else:
            self.Q_state_action[sa] = v
            self.Visits_state_action[sa] = 1

        self.Visits_state[stateKey] += 1
        visited.discard(stateKey)
        return v  # Return the value to propagate up the tree
