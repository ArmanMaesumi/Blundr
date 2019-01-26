# Blundr

Blundr is a chess AI that predicts the side advantage of a given chess board using deep learning. The current working model can be found in ```/models```. To train your own model, use ```train.py``` with a .npy dictionary file of type ```{'FEN':CP score}```. Sample data for ```2,700,000``` board states, and their evaluations are provided in ```/data```. To create your own training data, use ```stockfish_uci.py```. This script generates the ```.npy``` dictionary file given a pgn. To interface with the trained models and make predictions, use Blundr.py. The Blundr scripts accepts a chess FEN string as a board input.

Example using Blundr.py:
```
[Input]
Board FEN: 8/P7/1np2R1N/3k2B1/1p1P4/P2P3p/5PPP/6K1 w - - 1 37

[Output]
. . . . . . . .
P . . . . . . .
. n p . . R . N
. . . k . . B .
. p . P . . . .
P . . P . . . p
. . . . . P P P
. . . . . . K .
Prediction: [[ 0.44967598]]
White is favored to win.
