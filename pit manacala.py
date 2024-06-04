import Arena
from MCTS import MCTS
from mancala.MancalaGame import MancalaGame
from mancala.MancalaPlayers import *
from mancala.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = MancalaGame(6)

# all players
rp = RandomPlayer(g).play
gp = GreedyMancalaGamePlayer(g).play
hp = HumanMancalaGamePlayer(g).play

n1p = HumanMancalaGamePlayer(g).play
player2 = HumanMancalaGamePlayer(g).play

# # nnet players
# n1 = NNet(g)
# # n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
# args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
# mcts1 = MCTS(g, n1, args1)
# n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# if human_vs_cpu:
#     player2 = hp
# else:
#     n2 = NNet(g)
#     n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
#     args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
#     mcts2 = MCTS(g, n2, args2)
#     n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

#     player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=MancalaGame.display)

print(arena.playGames(2, verbose=True))