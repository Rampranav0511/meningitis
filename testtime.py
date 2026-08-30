import numpy as np
import chess
from chessfens import encod  # fens not needed here anymore

loading = np.load("neural network.npz")
# here loading is a dict-like container
"""this thing basically stored all our weights matrices as values to a key , for eg :
something like - "arr_0":first weights/first bias matrix,  depending on how we saved it earlier

in the saving process,  whole list wise saving wont work , so we need to unpack all the elements in
the weights and bias list and then save them in the .npz file , which is basically a compressed file"""

"""wdym by dict like container , its just basically a mapping ,a python implementation where dict was built and
it uses a hash table to look up values and stuff , but the implementation which numpy has is that it looks for the 
key's corresponding entry inside the zip archive, decompress those bytes, and reconstruct a numpy array from them"""
# our zip archive here is .npz file
"""loaded.files is basically the keys array which is self indexed , something like , arr_0 , arr_1 , stuff like that , 
        so now, we need to sort this as numpy doesnt really guarantee a sorted list even though the entry was done in that manner
its just a fallback check , so we split the key using the underscore as the delimiter(whatever u call ts) and sort checking the
number after the underscore"""
hahakey = sorted(loading.files, key=lambda k: int(k.split('_')[1]))
weights = []
biases = []
layers = len(hahakey) // 2

for i in range(layers):
    weights.append(loading[hahakey[i]])
    biases.append(loading[hahakey[i + layers]])

K = 0.00368208  # missing before -- same constant as sockfish.py's cp_win


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(x, weights, biases):
    act = x
    for i in range(layers):
        act = weights[i] @ act + biases[i]
        act = sigmoid(act)
    return act


def evaluate_fen(fen, weights, biases):
    board = chess.Board(fen)
    planes = encod(board)
    x = planes.flatten().reshape(-1, 1)
    score = forward(x, weights, biases)
    return score.item()


def win_prob_to_cp(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    cp = np.log(p / (1 - p)) / K
    return cp


if __name__ == "__main__":
    fen = input("enter a FEN: ").strip()
    win_prob = evaluate_fen(fen, weights, biases)
    cp = win_prob_to_cp(win_prob)
    pawns = cp / 100

    print(f"win probability (white): {win_prob:.4f}")
    print(f"eval: {pawns:+.2f}")


