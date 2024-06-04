"""
Author: Jayadev H
Date: 03/06/2024

                        Player 1
Player 1 Mancala ---> (0)(4)(4)(4)(4)(4)(4)
                      (4)(4)(4)(4)(4)(4)(0) <--- Player -1 Mancala
                        Player -1
"""


class Board:

    __player_to_row = {1:0,-1:1}

    def __init__(self, pocket_per_row = 6, stones_per_pocket = 4):
        "Set up initial board configuration."

        self.pocket_per_row = pocket_per_row
        self.stones_per_pocket = stones_per_pocket
        # Create the empty board array.
        self.player_count = 2
        self.pieces = [None] * self.player_count
        for i in range(self.player_count):
            self.pieces[i] = [self.stones_per_pocket] * (self.pocket_per_row + 1)

        self.pieces[0][0] = 0
        self.pieces[1][self.pocket_per_row] = 0

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def countDiff(self, player):
        """Counts the # pieces of the given player
        (1 for white, -1 for black, 0 for empty spaces)"""
        if player == 1:
            return self.pieces[0][0] - self.pieces[1][self.pocket_per_row]
        return self.pieces[1][self.pocket_per_row] - self.pieces[0][0]
        
    def end_game(self):
        sum_player_0 = sum(self.pieces[0])
        sum_player_1 = sum(self.pieces[1])
        self.pieces = [0] * self.player_count
        for i in range(self.player_count):
            self.pieces[i] = [0] * (self.pocket_per_row + 1)
        self.pieces[0][0] = sum_player_0
        self.pieces[1][self.pocket_per_row] = sum_player_1
            
    def get_legal_moves(self, player):
        """1 for each pocket that is positive, 0 for rest
        """
        print("printing the board to check legals move for player",player)
        print(self.pieces)
        
        if player == 1:
            print([1 if count > 0 else 0 for count in self.pieces[0][1:]])
            return [1 if count > 0 else 0 for count in self.pieces[0][:0:-1]]
        else:
            print([1 if count > 0 else 0 for count in self.pieces[1][:self.pocket_per_row]])
            return [1 if count > 0 else 0 for count in self.pieces[1][:self.pocket_per_row]]

    def has_legal_moves(self, player):
        return sum(self.get_legal_moves(player)) > 0

    def execute_move(self, move, player):
        """Perform the given move on the board;"""

        # Add the piece to the empty square.
        # print(move)

        # (0)(4)(4)(4)(4)(4)(4)
        # (4)(4)(4)(4)(4)(4)(0)
        if player == 1: #if first player his col is move+1
            col = move + 1
        else:
            col = move
        
        # print("move",player,move)
        # print(self.pieces)
        row = self.__player_to_row[player]
        stones = self[row][col]
        if stones == 0:
            raise Exception("Illigal move to take 0 stones")

        self[row][col] = 0
        self.add_stones(player, (row, col), stones)

    def add_stones(self, player, position, stones_left):
        if stones_left == 0:
            return
        row, col = position
        if row == 0:
            if col > 1 or (col == 1 and player == 1): #either above first mancala or first player and about to put in their own mancala
                self[0][col - 1] += 1
                return self.add_stones(player, (0, col - 1), stones_left - 1)
            
            # From 0th row to 1st row and add stone bot left
            self[1][0] += 1
            return self.add_stones(player, (1, 0), stones_left - 1)

        # (0)(4)(4)(4)(4)(4)(4)
        # (4)(4)(4)(4)(4)(4)(0)
        #  0  1  2  3  4  5  6
        if row == 1:
            if col < self.pocket_per_row-1 or (
                col == self.pocket_per_row-1 and player == -1
            ):
                self[1][col + 1] += 1
                return self.add_stones(player, (1, col + 1), stones_left - 1)
            # From 1st row to 0th row and add stone top right
            self[0][self.pocket_per_row] += 1
            return self.add_stones(player, (0, self.pocket_per_row), stones_left - 1)
