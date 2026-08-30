from datasets import load_dataset
import chess


import math
data=load_dataset("Lichess/chess-position-evaluations",split="train",data_files={"train": "data/data_0000.parquet"})

n=5000
subset=data.select(range(n)) #gets one small slice
label=[]


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

def cp_win(cp):
    return 0.5 + 0.5*(2/(1 + math.exp(-0.00368208*cp))-1)
for each in subset:
    fen=each["fen"]
    cp=each["cp"]
    mate=each["mate"]
    if mate is not None:
        y= 1.0 if mate>0 else 0.0
    elif cp is not None:
        
        y = cp_win(cp)   
    else:
        continue
    board=chess.Board(fen)
    x=encode_board(board).flatten()
    label.append((x,y))

split=int(len(label)*0.8)

valdata=label[split:]
tr=label[:split]