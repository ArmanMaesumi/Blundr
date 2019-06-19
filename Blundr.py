import numpy as np
import chess
import tensorflow as tf
import argparse
import os
from parse_FEN import FEN_to_one_hot
from train import str2bool

model = None


def main():
    # Request FEN string, then make prediction
    while 1:
        FEN = input('Board FEN: ')
        if FEN.lower() is 'stop' or FEN.lower() is 'quit':
            break

        test_board = chess.Board(FEN)
        # if not test_board.turn:
        #     FEN = test_board.mirror().fen()
        # print(find_best_move(test_board))
        board_arr, attack_matrix, turn = FEN_to_one_hot(FEN, None, None)
        current_score = predict(board_arr, attack_matrix, turn)
        print('Score: ' + str(current_score))
        print('Avg of neighboring boards: ' + str(avg_score_of_legal_moves(test_board)))
        find_best_move(test_board)
        find_min_loss(test_board, 0)


def find_min_loss(board, curr_score):
    highest_avg = -100000
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        avg_score = avg_score_of_legal_moves(board)
        if avg_score > highest_avg:
            highest_avg = avg_score
            best_move = move
        board.pop()

    board.push(best_move)
    print(best_move)


def avg_score_of_legal_moves(board):
    score_sum = 0
    possible_moves = 0
    for move in board.legal_moves:
        board.push(move)
        board_arr, attack_matrix, turn = FEN_to_one_hot(board.fen())
        score = predict(board_arr, attack_matrix, turn)
        score_sum += score
        possible_moves += 1
        board.pop()

    return (score_sum * 1.0) / possible_moves


def find_best_move(board):
    predictions = np.zeros(board.legal_moves.count())
    moves = []
    index = 0

    for move in board.legal_moves:
        board.push(move)

        prediction = test(board, 1)
        if np.isnan(prediction):
            prediction = 0

        predictions[index] = prediction
        board.pop()
        moves.append(move)
        index += 1

    print(np.argmax(predictions))
    print(moves[np.argmax(predictions)])
    print(moves[np.argmin(predictions)])
    return np.amin(predictions)


def test(board, depth):
    if depth == 0:
        board_arr, attack_matrix, turn = FEN_to_one_hot(board.fen(), None, None)
        return predict(board_arr, attack_matrix, turn)

    predictions = np.zeros(board.legal_moves.count())
    index = 0
    for move in board.legal_moves:
        board.push(move)
        predictions[index] = test(board, depth-1)
        board.pop()
        index += 1

    # print(predictions)
    return np.average(predictions)


def predict(board_arr, attack_matrix, turn):
    board_arr = np.asarray([board_arr])
    attack_matrix = np.asarray([attack_matrix])
    attack_matrix = tf.keras.utils.normalize(attack_matrix)
    absolute_prediction = model.predict([board_arr, attack_matrix])

    if turn:
        prediction = absolute_prediction[0][0]
    else:
        prediction = 1 - absolute_prediction[0][0]

    return prediction


if __name__ == '__main__':
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
