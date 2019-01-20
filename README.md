# Blundr

Blundr is a chess AI that predicts the centipawn score of a given chess board using machine learning. There is a model for white-to-play, and black-to-play in ```/models```. To train your own model, use ```train.py``` with a pgn file of the training matches, and a csv file with the centipawn evaluations at every move in the games. Sample data for ```50,000``` games and their evaluations are provided in ```/data```. To create your own training data, use ```stockfish_uci.py```. This script generates the csv file given the pgn.
