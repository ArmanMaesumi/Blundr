# Blundr

Blundr is a chess AI that predicts the side advantage of a given chess board using deep learning. The current working model can be found in ```/models```. To train your own model, use ```train.py``` with a .npy dictionary file of type ```{'FEN':CP score}```. Sample data for ```2,700,000``` board states, and their evaluations are provided in ```/data```. To create your own training data, use ```stockfish_uci.py```. This script generates the ```.npy``` dictionary file given a pgn. To interface with the trained models and make predictions, use Blundr.py. The Blundr scripts accepts a chess FEN string as a board input.

Example using Blundr.py:
```
[Input]
Board FEN: r1bk3r/bp1pnpp1/3N2q1/p1p1P2p/P6P/2P2Q2/1P3PP1/R1B1KB1R b KQ - 3 16

[Output]
r . b k . . . r
b p . p n p p .
. . . N . . q .
p . p . P . . p
P . . . . . . P
. . P . . Q . .
. P . . . P P .
R . B . K B . R
Prediction: [[ 0.91679049]]
White is favored to win.
```

## Accuracy

Currently, Blundr is trained on ~2,200,000 board states, and has been tested (out of sample) on ~400,000 board states. Across the entire test set, Blundr correctly predicts the side advantage roughly 75-78% of the time. However, when evaluating boards that are in the midgame to lategame, Blundr produces ~85-87% accuracy. This is because early game positions are harder to evaluate, as there is less information to work with (pieces lost, territory, etc).

## Dataset

The 2,700,000 board states provided in ```/data``` were evaluated using Stockfish on a i7-5820K @ 3.30 GHz. If the board was in its first 5 moves, Stockfish was allotted ```1 second``` to evaluate the position. After the 5th move Stockfish was given half a second. ```/data/known_scores.npy``` served as a hash table for all previously evaluated board positions. 

## Future Plans

1. Incorporate a matrix representing which pieces are under attack. 
2. Implement a CNN.
3. Predict centipawn score (magnitude of side advantage), rather than just side advantage.
