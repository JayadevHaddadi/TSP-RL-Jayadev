'''
Author: Jayadev H
Date: 03/06/2024


Terminology:
1. Pockets - Holes where stones can be places
2. Rows - Usually 1 row each, 2 in total
3. Stones - Each pocket can have stones
4. Mancalas - Ends, each player has one manchala, the one on their right

Start position:
(0)(4)(4)(4)(4)(4)(4)
(4)(4)(4)(4)(4)(4)(0)
'''

from __future__ import print_function
import sys

sys.path.append("..")
from Game import Game
from .MancalaLogic import Board
import numpy as np


class MancalaGame(Game):

    def __init__(self, pockert_per_row = 6, starting_stones_per_pocket = 4, row_count = 2):
        self.pockert_per_row = pockert_per_row
        self.starting_stones_per_pocket = starting_stones_per_pocket
        self.row_count = row_count

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.pockert_per_row, self.starting_stones_per_pocket)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.row_count, self.pockert_per_row)

    def getActionSize(self):
        # return number of actions
        return self.pockert_per_row

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action >= self.pockert_per_row:
            raise Exception("Action out of range")
        b = Board(self.pockert_per_row,0)
        b.pieces = np.copy(board)
        extra_move = b.execute_move(action, player)
        if extra_move:
            return (b.pieces, player) 
        else:
            return (b.pieces, -player) 

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = Board(self.pockert_per_row)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        return np.array(legalMoves)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost or draw
        # player = 1
        b = Board(self.pockert_per_row)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player) and b.has_legal_moves(-player):
            return 0
        b.end_game()
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        
        # print(f"WE GOT BOARD with player {player}")
        # print(board)
        if player == 1:
            return board
        return np.array([row[::-1] for row in board[::-1]]) #removed the array so it is same one, not a new one

    def getSymmetries(self, board, pi):
        # 180 rotation only
        assert len(pi) == self.pockert_per_row
        pi_board = pi[::-1]
        l = [(board, pi), (self.getCanonicalForm(board,-1), pi_board)]

        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(
            self.square_content[square] for row in board for square in row
        )
        return board_s

    def getScore(self, board, player):
        b = Board(self.pockert_per_row)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        row = board.shape[0]
        col = board.shape[1]

        print("     ", end="")
        for y in range(col-2,-1,-1):
            print(y, end="  ")
        print("")
        print("-------------------------")
        print("|  |", end="")  # print the row #
        for x in range(1,col):
            piece = board[0][x]  # get the piece to print
            print(f"{piece:2}", end=" ")
        print("|  | <-- Player  1")

        print(f"|{board[0][0]:2}|                  |{board[1][col-1]:2}|")

        print("|  |", end="")  # print the row #
        for x in range(col-1):
            piece = board[1][x]  # get the piece to print
            print(f"{piece:2}", end=" ")
        print("|  | <-- Player -1")

        print("-------------------------")
        
        print("     ", end="")
        for y in range(col-1):
            print(y, end="  ")
        print()
