import numpy as np
import chess
import tensorflow as tf
from train import board_to_array

# Load Keras models for white-to-play, and black-to-play
white_model = tf.keras.models.load_model('models/chess_model_white.model')
black_model = tf.keras.models.load_model('models/chess_model_black.model')


def main():
    # Request FEN string, then make prediction
    while 1:
        FEN = input('Board FEN: ')
        board = chess.Board(FEN)
        board_arr = board_to_array(board)
        print(board_arr)
        print(board)
        predict(board_arr, board.turn)


def predict(board_arr, turn):
    board_arr = [board_arr]
    model_input = np.asarray(board_arr)
    if turn:
        prediction = white_model.predict(model_input)
    else:
        prediction = black_model.predict(model_input)

    print("Prediction: " + str(prediction))
    if prediction[0] <= 0.5:
        print("White is favored to win.")
    else:
        print("Black is favored to win.")


if __name__ == '__main__':
    main()
