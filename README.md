## Chess Evaluation Neural Network

A chess position evaluator built entirely from scratch in NumPy. Every forward pass, every gradient, every weight update was derived and implemented by hand. No pre-existing frameworks used.

Feed it a FEN string. Get back an evaluation — the same idea as the eval bar on Lichess or Chess.com.

## How a position becomes a number

**Encoding the board.** A chess position gets turned into 12 binary 8x8 grids — one grid per piece type per color (white pawns, white knights, all the way to black king). Flatten those together and you get a 768-length vector, which is all the network ever sees. No material count, no positional heuristics handed to it — it has to learn everything itself from where the pieces are sitting.

**The network itself.** A plain fully-connected MLP. Every layer, hidden and output alike, runs through a sigmoid:

```
a(l) = sigmoid( W(l) . a(l-1) + b(l) )
```

**Training it.** Backprop, done manually — the error at the output gets pushed backward through each layer, the gradient for every weight and bias worked out via the chain rule through the sigmoid's derivative, and every weight nudged a small step in the direction that reduces the error. Trained in mini-batches, with a validation set checked every epoch and early stopping once validation loss stops improving — so what actually gets saved is the best version of the network, not just whatever epoch it happened to stop on.


## Where the training data comes from

Labels come from the [Lichess position evaluations dataset](https://huggingface.co/datasets/Lichess/chess-position-evaluations) on Hugging Face — millions of real positions, each already scored by Stockfish. Positions get pulled, encoded into the 12-plane format, converted to win-probability targets, and split 80/20 into training and validation sets.

## What's in the repo

| File | What it does |
|---|---|
| `sockfish.py` | Pulls from the Lichess dataset, encodes positions, converts scores into win-probability labels, builds the train/val split |
| `neural.py` | The actual training script — builds the network, runs the training loop, saves the best weights to `neural network.npz` |
| `testtime.py` | Load the trained weights, hand it a FEN, get back a win probability and a converted eval |
| `chessfens.py` | Shared board-encoding function, plus a helper for pulling FENs out of a `.pgn` file |
| `apicom.py` | Grabs a player's game history from the Chess.com API into a `.pgn` file — a standalone utility right now, not yet feeding into training |


