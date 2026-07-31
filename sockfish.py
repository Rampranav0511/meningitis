import chess
import chess.engine
import numpy as np

 

def encode_board(board):
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


x=[]
y=[]
with open("pgn_games","r") as f:
    while True:
        game=chess.pgn.read_game(f) # here we are creating a game object
        if game is None:
            break
        br=game.board() # this function also returns an chess.board object
        for move in game.mainline_moves: # game.mainline_moves gives an iterable class , but here game.mainline

            br.push(move) # this just moved a piece
            info=engine.analyse(br,chess.engine.Limit(depth=11))
            score=info["score"].white().score(mate_score=10000)

            x.append(encode_board(br)) """ so basically br is a chess.Board object which basically when printed, python
            automatically converts and prints it as a string. it shows the board position"""
                     
        """but neural network only takes in numbers or vectors/arrays , so we need to 
        encode this board into a 3d numerical array"""
            y.append(score)
