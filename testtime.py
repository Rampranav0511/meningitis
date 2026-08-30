import numpy as np
import chess
from chessfens import fens,encod

loading=np.load("neural network.npz")
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
hahakey=sorted(loading.files,key=lambda k: int(k.split('_')[1]))
weights=[]
biases=[]
layers=len(hahakey)//2 # how many weights and bias matrices exist

for i in range(layers):

    weights.append([loading[hahakey[i]]])
    biases.append([loading[hahakey[i+layers]]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(x, weights, biases):
    
    act = x
    for i in range(layers):
        act = weights[i] @ act + biases[i]
        act = sigmoid(act)
    return act


def evaluate_fen(fen, weights, biases):
    """
    Takes a FEN string, encodes it the same way training data was encoded,
    runs it through the network, and returns a win-probability score (0-1).
    """
    board = chess.Board(fen)
    planes = encod(board)                  # (12, 8, 8)
    x = planes.flatten().reshape(-1, 1)    # (768, 1) column vector
    score = forward(x, weights, biases)
    return score.item()

for i in range(len(fens)):
    score=evaluate_fen(fens[i],weights,biases)




