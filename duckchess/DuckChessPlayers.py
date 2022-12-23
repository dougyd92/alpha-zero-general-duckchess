import numpy as np

from .DuckChessLogic import KnightMoves, Directions

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanDuckChessPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)

        print('Enter a move in this format: Rank File Direction NumK DRank DFile')
        print('where Rank,File are the coordinates of the piece to move,')
        print('where DRank,DFile are where to place the duck.')
        print('For a queen-style move, NumK is the number of spaces to move,')
        print('and Direction is one of N NE E SE S SW W NW.')
        print('For a knight-style move, NumK should be K,')
        print('and Direction is one of NNE NEE SEE SSE SSW SWW NWW NNW')

        while True:
            input_move = input().split(" ")
            if len(input_move) == 6:
                try:
                    rank = int(input_move[0])
                    file = int(input_move[1])
                    direction = input_move[2].upper()
                    numk = input_move[3]
                    duckrank = int(input_move[4])
                    duckfile = int(input_move[5])

                    move_type = 0
                    if numk.upper() == 'K':
                        for k in KnightMoves:
                            if direction == k.name:
                                move_type = k
                    else:
                        amount = int(numk)
                        direction_enum = None
                        for d in Directions:
                            if direction == d.name:
                                direction_enum = d
                        move_type = board.getRelativeMoveIndex(direction_enum, amount)

                    a = 37376 * rank + 4672 * file + 64 * move_type + 8 * duckrank + duckfile
                    if valid[a]:
                        break
                except ValueError:
                    'Invalid move'
            print('Invalid move')
        return a
