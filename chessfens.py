import chess.pgn
import numpy as np
fens = []

with open("mygames.pgn") as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break  # reached end of file

        board = game.board()  # starting position for this game
        for move in game.mainline_moves():
            board.push(move)
            fens.append(board.fen())
         
inp=[]
            
def encod(board):
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    piece_to_plane = {
        (chess.PAWN,   chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2,
        (chess.ROOK,   chess.WHITE): 3,
        (chess.QUEEN,  chess.WHITE): 4,
        (chess.KING,   chess.WHITE): 5,
        (chess.PAWN,   chess.BLACK): 6,
        (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.ROOK,   chess.BLACK): 9,
        (chess.QUEEN,  chess.BLACK): 10,
        (chess.KING,   chess.BLACK): 11,
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane_idx = piece_to_plane[(piece.piece_type, piece.color)]
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            planes[plane_idx, row, col] = 1.0

    return planes      
for i in range(len(fens)):

    boardgame=chess.Board(fens[i])
    xxx=encod(boardgame).flatten() 
    inp.append(xxx)         
