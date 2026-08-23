# ♟️ Chess Evaluation Neural Network

**A chess position evaluator built entirely from scratch — no PyTorch, no TensorFlow, no autograd. Just NumPy and the math, implemented by hand.**

Feed it a FEN string. Get back an evaluation score — the same idea as the eval bar on Lichess or Chess.com, except every gradient here was derived and coded manually.

---

## Why this exists

Most "neural network from scratch" projects stop at MNIST digit classification. This one goes further: a full regression MLP — forward pass, backpropagation, loss computation, and training loop — built without any autodiff framework, trained to approximate Stockfish's positional judgment from raw board state.

No `loss.backward()`. No hidden autograd graph doing the calculus for you. Every derivative in this network was worked out by hand and translated into NumPy matrix operations.

## How it works

```
FEN string  →  12-plane binary board encoding  →  MLP  →  centipawn evaluation
```

- **Input representation:** Each board is encoded as 12 binary 8×8 planes — one per piece type per color (768 input features total)
- **Architecture:** Multi-layer perceptron regression model, trained via backpropagation and gradient descent implemented from first principles
- **Labels:** Stockfish-generated centipawn evaluations, sourced from real games pulled via the Lichess/Chess.com APIs
- **Training:** MSE loss, early stopping with patience, validation loss tracking to avoid overfitting

What's implemented from scratch

Component  Status 

| Forward propagation
| Backpropagation (manual chain rule)
| MSE loss 
| Gradient descent weight updates 
| Early stopping (patience-based)
| Validation loss tracking
| Board → tensor encoding (768-dim) 

## Quickstart

```bash
git clone <your-repo-url>
cd chess-eval-nn
pip install -r requirements.txt

python evaluate.py --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
```

```
Output: +0.35 (White is slightly better)
```

## Project structure

```
├── model/           # From-scratch MLP: forward pass, backprop, training loop
├── data/             # Board encoding + Stockfish-labeled dataset pipeline
├── chess.py          # Fetches games via the Chess.com API
├── evaluate.py        # CLI: FEN in, eval score out
└── notebooks/        # Training experiments and loss curves
```

## Why build it this way

Frameworks like PyTorch abstract away the exact mechanics of how a network learns. Building this by hand meant working through the calculus of backpropagation, debugging gradient flow issues with nothing but print statements and math, and understanding *why* a network converges rather than trusting that it will.

## Roadmap

- [ ] Expand training set with deeper Stockfish search depth for label quality
- [ ] Experiment with deeper architectures / alternative encodings
- [ ] Web demo — paste a FEN, see the eval bar move
- [ ] Benchmark against Stockfish evaluations on held-out positions

## Tech

`Python` · `NumPy` · `python-chess` · `Stockfish`

---

*Built as a from-scratch deep dive into neural network fundamentals — every layer of math implemented by hand before scaling up.*
