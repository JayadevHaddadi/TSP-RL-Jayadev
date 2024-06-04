
from mancala.MancalaGame import MancalaGame as Game

g = Game(6,4)
board = g.getInitBoard()
print(board)
board_new, player = g.getNextState(board=board,player=0,action=0)
print(board_new)
print(g.getValidMoves(board,0))
print(g.getValidMoves(board,1))
print(g.getCanonicalForm(board_new,0))
print(g.getCanonicalForm(board_new,1))


