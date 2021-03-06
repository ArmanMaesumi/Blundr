# Blundr

Blundr is a deep neural network that predicts the side advantage of a given chess board. The current working model can be found in [releases](https://github.com/ArmanMaesumi/Blundr/releases). To train/test a model, use ```MLP.py``` with a .npy dictionary file of type ```{'FEN':CP score}```. Sample data for ```2,700,000``` board states, and their evaluations are provided in ```/data```. Please send me an email If you'd like to use the full dataset of ~```20,000,000``` board states. To create your own training data, use ```stockfish_eval.py```. This script generates the ```.npy``` dictionary file given a pgn. To interface with the trained models and make predictions, use Blundr.py. The Blundr scripts accepts a chess FEN string as a board input.

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
## The Model and Its Performance

Blundr uses two input layers, one which is a one-hot encoded chess board, and the other representing the tiles that are under attack by white/black/both. The input layers feed into two separate deep layers, which are then concatenated and fed through 3 additional deep layers. The final output layer has 3 neurons for each class (white winning, black winning, draw).

Currently, Blundr is trained on ~3,000,000 board states, and has been tested (out of sample) on ~250,000 board states. Across the entire test set, Blundr correctly predicts the side advantage 89% of the time. However, when evaluating boards that have a substantial advantage, Blundr produces ~96% accuracy.

### Pre-trained model

The current pre-trained models can be found in [releases](https://github.com/ArmanMaesumi/Blundr/releases)

## Dataset

The 2,700,000 board states provided in ```/data``` were evaluated using Stockfish on a i7-5820K @ 3.30 GHz. If the board was in its first 5 moves, Stockfish was allotted ```1 second``` to evaluate the position. After the 5th move Stockfish was given half a second. ```/data/known_scores.npy``` served as a hash table for all previously evaluated board positions. 
