# Blundr

Blundr is a chess AI that predicts the centipawn score of a given chess board using machine learning. There is a model for white-to-play, and black-to-play in ```/models```. To train your own model, use ```train.py``` with a pgn file of the training matches, and a csv file with the centipawn evaluations at every move in the games. Sample data for ```50,000``` games and their evaluations are provided in ```/data```. To create your own training data, use ```stockfish_uci.py```. This script generates the csv file given the pgn. To interface with the trained models and make predictions, use Blundr.py. The Blundr scripts accepts a chess FEN string as a board input.

Example using Blundr.py:
```
[Input]
Board FEN: 8/P7/1np2R1N/3k2B1/1p1P4/P2P3p/5PPP/6K1 w - - 1 37

[Output]
[[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 6, 2, 0, 0, 3, 0, 5], [0, 0, 0, 12, 0, 0, 7, 0], [0, 2, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 2], [0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 11, 0]]
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
