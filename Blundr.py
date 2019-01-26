import numpy as np
import chess
import tensorflow as tf
import argparse
import os
from parse_FEN import FEN_to_one_hot
from train import str2bool

# Load Keras models for white-to-play, and black-to-play
# white_model = tf.keras.models.load_model('models/chess_model_white.model')
# black_model = tf.keras.models.load_model('models/chess_model_black.model')
model = None


def main():
    # Request FEN string, then make prediction
    while 1:
        FEN = input('Board FEN: ')
        if FEN.lower() is 'stop' or FEN.lower() is 'quit':
            break

        board = chess.Board(FEN)
        board_arr = FEN_to_one_hot(FEN)
        print(board_arr)
        print(board)
        predict(board_arr, board.turn)


def predict(board_arr, turn):
    board_arr = [board_arr]
    model_input = np.asarray(board_arr)
    if turn:
        prediction = model.predict(model_input)
    else:
        prediction = model.predict(model_input)

    print("Prediction: " + str(prediction))
    if prediction[0] >= 0.5:
        print("White is favored to win.")
    else:
        print("Black is favored to win.")


if __name__ == '__main__':
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of .model file.')
    parser.add_argument('--gpu', type=str2bool, nargs='?', help='(T/F) use GPU.')
    args = parser.parse_args()

    if args.gpu:
        # Set Tf session to use GPU memory growth
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)
    else:
        # Disable GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not str(args.model_name).endswith('.model'):
        model_name = str(args.model_name) + '.model'
    else:
        model_name = str(args.model_name)

    model = tf.keras.models.load_model(model_name)
    main()
