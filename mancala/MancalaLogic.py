"""
Author: Jayadev H
Date: 03/06/2024
"""


class Board:
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

    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        if color == 0:
            return self.pieces[0][0] - self.pieces[1][self.pocket_per_row]
        return self.pieces[1][self.pocket_per_row] - self.pieces[0][0]
    
    def count_final_score(self, color):
        if color == 0:
            return sum
        
    def end_game(self):
        sum_player_0 = sum(self.pieces[0])
        sum_player_1 = sum(self.pieces[1])
        self.pieces = [0] * self.player_count
        for i in range(self.player_count):
            self.pieces[i] = [0] * (self.pocket_per_row + 1)
        self.pieces[0][0] = sum_player_0
        self.pieces[1][self.pocket_per_row] = sum_player_1
            
    def get_legal_moves(self, color):
        """1 for each pocket that is positive, 0 for rest
        """
        if color == 0:
            return [1 if count > 0 else 0 for count in self.pieces[color][1:]]
        else:
            return [1 if count > 0 else 0 for count in self.pieces[color][:self.pocket_per_row]]

    def has_legal_moves(self, color):
        return sum(self.get_legal_moves(color)) > 0

    def add_stones(self, player, position, stones_left):
        if stones_left == 0:
            return
        row, col = position
        if row == 0:
            if col > 1 or (col == 1 and player == 0):
                self[0][col - 1] += 1
                return self.add_stones(player, (row, col - 1), stones_left - 1)
            # From 0th row to 1st row, either on col==0 or on col 1 but other player
            self[1][0] += 1
            return self.add_stones(player, (1, 0), stones_left - 1)

        if row == 1:
            if col < self.pocket_per_row-2 or (
                col == self.pocket_per_row-1 and player == 1
            ):
                self[1][col + 1] += 1
                return self.add_stones(player, (row, col + 1), stones_left - 1)
            # From 1st row to 0th row
            self[0][self.pocket_per_row] += 1
            return self.add_stones(player, (1, 0), stones_left - 1)

    def execute_move(self, move, color):
        """Perform the given move on the board;"""

        # Add the piece to the empty square.
        # print(move)

        # (0)(4)(4)(4)(4)(4)(4)
        # (4)(4)(4)(4)(4)(4)(0)
        if color == 0:
            move += 1
        
        print("move",color,move)
        print(self.pieces)
        stones = self[color][move]
        if stones == 0:
            raise Exception("Illigal move to take 0 stones")

        self[color][move] = 0
        self.add_stones(color, (color, move), stones)