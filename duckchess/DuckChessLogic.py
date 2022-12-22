import numpy as np
from enum import IntEnum

# 1 binary layer for duck piece
# 6 binary layers for white pieces
# 1 uniform binary layer for current player
# 6 binary layers for black pieces
# 4 uniform binary layers for castle eligibility
NUM_PLANES = 18

# 8x8 for which piece to move, 
# 73 for how that piece is being moved,
# 8x8 for the next duck location
ACTION_SIZE = 299008

# Layers for pieces are 6 minus the piece encoding
# e.g. current player's pawns are 6-1=layer 5
# opponent's queen is 6-(-6)=layer12
# So Layer 0 has the neutral duck
# Layers 1-6 are the current player's pieces
# Layers 8-13 are the opponent's pieces
# Layer 7 is used to denote the current player's color
DUCKER_LAYER = 0
PLAYER_LAYER = 7
PLAYER_QUEENSIDE_CASTLE_LAYER = 14
PLAYER_KINGSIDE_CASTLE_LAYER = 15
OPP_QUEENSIDE_CASTLE_LAYER = 16
OPP_KINGSIDE_CASTLE_LAYER = 17

class Pieces(IntEnum):
    PLAYER_P = 1
    PLAYER_R = 2
    PLAYER_N = 3
    PLAYER_B = 4
    PLAYER_K = 5
    PLAYER_Q = 6
    OPPONENT_P = -1
    OPPONENT_R = -2
    OPPONENT_N = -3
    OPPONENT_B = -4
    OPPONENT_K = -5
    OPPONENT_Q = -6
    DUCK = 7

class KnightMoves(IntEnum):
    NNW = 56
    NNE = 57
    NEE = 58
    SEE = 59
    SSE = 60
    SSW = 61
    SWW = 62
    NWW = 63

class Directions(IntEnum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7

class DuckChessBoard():
    def __init__(self):
        self.pieces = np.zeros((8,8), dtype='int8')
        self.pieces[6].fill(Pieces.PLAYER_P)
        self.pieces[7][0] = Pieces.PLAYER_R
        self.pieces[7][7] = Pieces.PLAYER_R
        self.pieces[7][1] = Pieces.PLAYER_N
        self.pieces[7][6] = Pieces.PLAYER_N
        self.pieces[7][2] = Pieces.PLAYER_B
        self.pieces[7][5] = Pieces.PLAYER_B
        self.pieces[7][3] = Pieces.PLAYER_Q
        self.pieces[7][4] = Pieces.PLAYER_K
        self.pieces[1].fill(Pieces.OPPONENT_P)
        self.pieces[0][0] = Pieces.OPPONENT_R
        self.pieces[0][7] = Pieces.OPPONENT_R
        self.pieces[0][1] = Pieces.OPPONENT_N
        self.pieces[0][6] = Pieces.OPPONENT_N
        self.pieces[0][2] = Pieces.OPPONENT_B
        self.pieces[0][5] = Pieces.OPPONENT_B
        self.pieces[0][3] = Pieces.OPPONENT_Q
        self.pieces[0][4] = Pieces.OPPONENT_K
        self.white_to_move = True
        self.player_can_castle_queenside = True
        self.player_can_castle_kingside = True
        self.opponent_can_castle_queenside = True
        self.opponent_can_castle_kingside = True
        #TODO move counts, repetition counts?, 50move, and en passant
    
    def encode(self):
        board = np.zeros((NUM_PLANES, 8, 8), dtype=bool)
        for rank in range(8):
            for file in range(8):
                piece = self.pieces[rank][file]
                layer = 6 - piece
                board[layer][rank][file] = True
        board[PLAYER_QUEENSIDE_CASTLE_LAYER].fill(self.player_can_castle_queenside)
        board[PLAYER_KINGSIDE_CASTLE_LAYER].fill(self.player_can_castle_kingside)
        board[OPP_QUEENSIDE_CASTLE_LAYER].fill(self.opponent_can_castle_queenside)
        board[OPP_KINGSIDE_CASTLE_LAYER].fill(self.opponent_can_castle_kingside)
        if self.white_to_move:
            board[PLAYER_LAYER].fill(True)
        else:
            board[PLAYER_LAYER].fill(False)

        return board
    
    def getValidMoves(self):
        moves = np.zeros((8, 8, 73, 8, 8), dtype=bool)
        for rank in range(8):
            for file in range(8):
                piece = self.pieces[rank][file]
                if piece <= 0 or piece == Pieces.DUCK:
                    continue
                elif piece == Pieces.PLAYER_P:
                    self.addPawnMoves(moves, rank, file)
                elif piece == Pieces.PLAYER_R:
                    self.addRookMoves(moves, rank, file)       
                elif piece == Pieces.PLAYER_B:
                    self.addBishopMoves(moves, rank, file)                          
                elif piece == Pieces.PLAYER_Q:
                    self.addRookMoves(moves, rank, file)
                    self.addBishopMoves(moves, rank, file)
                elif piece == Pieces.PLAYER_K:
                    self.addKingMoves(moves, rank, file)
                elif piece == Pieces.PLAYER_N:
                    self.addKnightMoves(moves, rank, file)                    
        return moves

    def addPawnMoves(self, moves, rank, file):
        # Todo: consider underpromotions. Currently will assume promoting to Queen
        # Todo en passant

        # Capture left
        if file > 0: 
            if self.pieces[rank-1][file-1] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.NW, 1)] = self.getPossibleDuckMoves(rank, file, rank-1, file-1)

        # Capture right
        if file < 7: 
            if self.pieces[rank-1][file+1] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.NE, 1)] = self.getPossibleDuckMoves(rank, file, rank-1, file+1)
        
        # Forward move
        if self.pieces[rank-1][file] == 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.N, 1)] = self.getPossibleDuckMoves(rank, file, rank-1, file)
            # Double forward move
            if rank == 6 and self.pieces[rank-2][file] == 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.N, 2)] = self.getPossibleDuckMoves(rank, file, rank-2, file)
    
    def addRookMoves(self, moves, rank, file):
        # Forward moves
        for i in range(rank):
            # Own piece or duck in the way
            if self.pieces[rank-1-i][file] > 0:
                break

            # Capture enemy piece
            if self.pieces[rank-1-i][file] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.N, 1+i)] = self.getPossibleDuckMoves(rank, file, rank-1-i, file)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.N, 1+i)] = self.getPossibleDuckMoves(rank, file, rank-1-i, file)

        # Backward moves
        for i in range(7 - rank):
            # Own piece or duck in the way
            if self.pieces[rank+1+i][file] > 0:
                break

            # Capture enemy piece
            if self.pieces[rank+1+i][file] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.S, 1+i)] = self.getPossibleDuckMoves(rank, file, rank+1+i, file)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.S, 1+i)] = self.getPossibleDuckMoves(rank, file, rank+1+i, file)


        # Left moves
        for i in range(file):
            # Own piece or duck in the way
            if self.pieces[rank][file-1-i] > 0:
                break

            # Capture enemy piece
            if self.pieces[rank][file-1-i] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.W, 1+i)] = self.getPossibleDuckMoves(rank, file, rank, file-1-i)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.W, 1+i)] = self.getPossibleDuckMoves(rank, file, rank, file-1-i)  

        # Right moves
        for i in range(7 - file):
            # Own piece or duck in the way
            if self.pieces[rank][file+1+i] > 0:
                break

            # Capture enemy piece
            if self.pieces[rank][file+1+i] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.E, 1+i)] = self.getPossibleDuckMoves(rank, file, rank, file+1+i)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.E, 1+i)] = self.getPossibleDuckMoves(rank, file, rank, file+1+i)                    

    def addBishopMoves(self, moves, rank, file):
        # NW moves
        for i in range(min(rank, file)):
            new_rank = rank-1-i
            new_file = file-1-i
            # Own piece or duck in the way
            if self.pieces[new_rank][new_file] > 0:
                break

            # Capture enemy piece
            if self.pieces[new_rank][new_file] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.NW, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.NW, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)

        # NE moves
        for i in range(min(rank, 7-file)):
            new_rank = rank-1-i
            new_file = file+1+i
            # Own piece or duck in the way
            if self.pieces[new_rank][new_file] > 0:
                break

            # Capture enemy piece
            if self.pieces[new_rank][new_file] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.NE, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.NE, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)

        # SE moves
        for i in range(min(7-rank, 7-file)):
            new_rank = rank+1+i
            new_file = file+1+i
            # Own piece or duck in the way
            if self.pieces[new_rank][new_file] > 0:
                break

            # Capture enemy piece
            if self.pieces[new_rank][new_file] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.SE, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.SE, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)    

        # SW moves
        for i in range(min(7-rank, file)):
            new_rank = rank+1+i
            new_file = file-1-i
            # Own piece or duck in the way
            if self.pieces[new_rank][new_file] > 0:
                break

            # Capture enemy piece
            if self.pieces[new_rank][new_file] < 0:
                moves[rank][file][self.getRelativeMoveIndex(Directions.SW, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)
                break
            
            # Nothing in the way, add move and keep going
            moves[rank][file][self.getRelativeMoveIndex(Directions.SW, 1+i)] = self.getPossibleDuckMoves(rank, file, new_rank, new_file)             

    def addKingMoves(self, moves, rank, file):
        if rank > 0 and file > 0 and self.pieces[rank-1][file-1] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.NW, 1)] = self.getPossibleDuckMoves(rank, file, rank-1, file-1)
        if rank > 0 and self.pieces[rank-1][file] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.N, 1)] = self.getPossibleDuckMoves(rank, file, rank-1, file)
        if rank > 0 and file < 7 and self.pieces[rank-1][file+1] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.NE, 1)] = self.getPossibleDuckMoves(rank, file, rank-1, file+1)
        if file < 7 and self.pieces[rank][file+1] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.E, 1)] = self.getPossibleDuckMoves(rank, file, rank, file+1)
        if rank < 7 and file < 7 and self.pieces[rank+1][file+1] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.SE, 1)] = self.getPossibleDuckMoves(rank, file, rank+1, file+1)
        if rank < 7 and self.pieces[rank+1][file] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.S, 1)] = self.getPossibleDuckMoves(rank, file, rank+1, file) 
        if rank < 7 and file > 0 and self.pieces[rank+1][file-1] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.SW, 1)] = self.getPossibleDuckMoves(rank, file, rank+1, file-1)
        if file > 0 and self.pieces[rank][file-1] <= 0:
            moves[rank][file][self.getRelativeMoveIndex(Directions.W, 1)] = self.getPossibleDuckMoves(rank, file, rank, file-1)
        # todo add castling

    def addKnightMoves(self, moves, rank, file):
        # Moves 0-55 are queen style moves
        # Moves 56-63 are knight style moves: NNW NNE NEE SEE SSE SSW SWW NWW
        if rank >= 2 and file >= 1 and self.pieces[rank-2][file-1] <= 0:
            moves[rank][file][KnightMoves.NNW] = self.getPossibleDuckMoves(rank, file, rank-2, file-1)

        if rank >= 2 and file <= 6 and self.pieces[rank-2][file+1] <= 0:
            moves[rank][file][KnightMoves.NNE] = self.getPossibleDuckMoves(rank, file, rank-2, file+1)

        if rank >= 1 and file <= 5 and self.pieces[rank-1][file+2] <= 0:
            moves[rank][file][KnightMoves.NEE] = self.getPossibleDuckMoves(rank, file, rank-1, file+2)

        if rank <= 6 and file <= 5 and self.pieces[rank+1][file+2] <= 0:
            moves[rank][file][KnightMoves.SEE] = self.getPossibleDuckMoves(rank, file, rank+1, file+2)

        if rank <= 5 and file <= 6 and self.pieces[rank+2][file+1] <= 0:
            moves[rank][file][KnightMoves.SSE] = self.getPossibleDuckMoves(rank, file, rank+2, file+1)

        if rank <= 5 and file >= 1 and self.pieces[rank+2][file-1] <= 0:
            moves[rank][file][KnightMoves.SSW] = self.getPossibleDuckMoves(rank, file, rank+2, file-1)

        if rank <= 6 and file >= 2 and self.pieces[rank+1][file-2] <= 0:
            moves[rank][file][KnightMoves.SWW] = self.getPossibleDuckMoves(rank, file, rank+1, file-2)

        if rank >= 1 and file >= 2 and self.pieces[rank-1][file-2] <= 0:
            moves[rank][file][KnightMoves.NWW] = self.getPossibleDuckMoves(rank, file, rank-1, file-2)

    # The output format for actions is 8x8x73x8x8
    # Of the 73, the first 56 are queen-style moves
    def getRelativeMoveIndex(self, direction, amount):
        return direction*7+amount-1
    
    def getPossibleDuckMoves(self, prev_rank, prev_file, next_rank, next_file):
        # Encode the 8x8 locations that the duck can be moved to next
        # This is all the currently empty spaces on the board,
        # minus the space being moved to,
        # plus the space that was just vacated
        duck_moves = (self.pieces == 0)
        duck_moves[prev_rank][prev_file] = True
        duck_moves[next_rank][next_file] = False
        return duck_moves
    
    def decodeAction(self, action_index):
        # Take the index of the flattened action tensor
        # and translate it back to the action being taken
        rank = action_index // 37376
        action_index = action_index - 37376 * rank
        file = action_index // 4672
        action_index = action_index - 4672 * file
        move_type = action_index // 64
        action_index = action_index - 64 * move_type
        duck_rank = action_index // 8
        duck_file = action_index - duck_rank * 8
        return (rank, file, move_type, duck_rank, duck_file)

    def decodeChessMove(self, move_type):
        # Queen-style move
        if move_type < 56:
            direction = move_type // 7
            amount = move_type % 7 + 1
            if direction == Directions.N:
                rank_direction = -1
                file_direction = 0
            elif direction == Directions.NE:
                rank_direction = -1
                file_direction = 1
            elif direction == Directions.E:
                rank_direction = 0
                file_direction = 1
            elif direction == Directions.SE:
                rank_direction = 1
                file_direction = 1                                   
            elif direction == Directions.S:
                rank_direction = 1
                file_direction = 0
            elif direction == Directions.SW:
                rank_direction = 1
                file_direction = -1
            elif direction == Directions.W:
                rank_direction = 0
                file_direction = -1
            elif direction == Directions.NW:
                rank_direction = -1
                file_direction = -1                             
            return (rank_direction * amount, file_direction * amount)
        # Knight-style move
        elif move_type < 64:
            if move_type == KnightMoves.NNW:
                return (-2, -1)
            elif move_type == KnightMoves.NNE:
                return (-2, +1)
            elif move_type == KnightMoves.NEE:
                return (-1, +2)
            elif move_type == KnightMoves.SEE:
                return (+1, +2)
            elif move_type == KnightMoves.SSE:
                return (+2, +1)
            elif move_type == KnightMoves.SSW:
                return (+2, -1)
            elif move_type == KnightMoves.SWW:
                return (+1, -2)
            elif move_type == KnightMoves.NWW:
                return (-1, -2)
        else:
            # TODO underpromotions
            raise Exception(f"Move type {move_type} not implemented yet")

    def performMove(self, action):
        rank, file, move_type, duck_rank, duck_file = self.decodeAction(action)

        # Which pieces is being moved?
        piece = self.pieces[rank][file]
        if piece <= 0:
            raise Exception("Trying to move a piece that's not yours")
        
        # Where is it being moved to?
        rank_offset, file_offset = self.decodeChessMove(move_type)
        new_rank = rank + rank_offset
        new_file = file + file_offset
        
        # Check for promotion
        if new_rank == 0 and piece == Pieces.PLAYER_P:
            piece == Pieces.PLAYER_Q

        # Move the piece
        self.pieces[rank][file] = 0
        self.pieces[new_rank][new_file] = piece

        # Negative bc we are about to flip the board to the other player's perspective
        self.pieces[duck_rank][duck_file] = -1 * Pieces.DUCK

        # Flip the board around now for other player's perspective
        # Note that this is technically mirrored, so that the 'images' 
        # that are inputs to the NN look the same for black and white
        # (ie queen is always on the left, unlike real chess)
        self.pieces = -1 * self.pieces
        self.pieces[[0,7]] = self.pieces[[7,0]]
        self.pieces[[1,6]] = self.pieces[[6,1]]
        self.pieces[[2,5]] = self.pieces[[5,2]]
        self.pieces[[3,4]] = self.pieces[[4,3]]

        self.white_to_move = not self.white_to_move
        self.player_can_castle_queenside, self.opponent_can_castle_queenside = self.opponent_can_castle_queenside, self.player_can_castle_queenside
        self.player_can_castle_kingside, self.opponent_can_castle_kingside = self.opponent_can_castle_kingside, self.player_can_castle_kingside
    
    def checkForGameOver(self):
        # todo stalemates
        # todo draw due to repetition or move limit
        if not np.any(self.pieces == Pieces.PLAYER_K):
            return 1
        if not np.any(self.pieces == Pieces.OPPONENT_K):
            # This shouldn't happen, but checking bc not sure which player is which
            raise Exception("Opponent already lost, you shouldn't have another turn")
        return 0
    
    def hashKey(self):
        return np.array2string(self.pieces) + str(self.white_to_move) + str(self.player_can_castle_queenside) \
            + str(self.player_can_castle_kingside) + str(self.opponent_can_castle_queenside) + str(self.opponent_can_castle_kingside)