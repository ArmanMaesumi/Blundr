import tensorflow as tf
import numpy as np
import time
from stockfish_uci import import_hash_table
from parse_FEN import FEN_to_one_hot
import argparse

# empty = 0
# w_pawn = 1
# b_pawn = 2
# w_rook = 3
# b_rook = 4
# w_knight = 5
# b_knight = 6
# w_bishop = 7
# b_bishop = 8
# w_queen = 9
# b_queen = 10
# w_king = 11
# b_king = 12
#
# piece_dict = {'.': empty, 'P': w_pawn, 'p': b_pawn, 'R': w_rook,
#               'r': b_rook, 'N': w_knight, 'n': b_knight, 'B': w_bishop,
#               'b': b_bishop, 'Q': w_queen, 'q': b_queen, 'K': w_king,
#               'k': b_king}


# # Create chess board from FEN string
# def board_arr_from_fen(FEN):
#     board = chess.Board(FEN)
#     return board_to_array(board)
#
#
# # Convert chess board to array using piece_dict values
# def board_to_array(board):
#     board_arr = []
#     board_str_arr = str(board).split("\n")
#     for row in board_str_arr:
#         curr_row = row.split(" ")
#         board_arr.append(list(map(piece_dict.get, curr_row)))
#
#     return board_arr
#
#
# def parse_stockfish_scores(num_scores):
#     stockfish_csv = 'data\\stockfish_scores.csv'
#     names = ['Event', 'MoveScores']
#     stockfish_scores = pandas.read_csv(stockfish_csv, names=names)
#     move_scores = []
#
#     # Read in stockfish evaluation for player moves
#     for i in range(num_scores):
#         if 0 < i:
#             curr_scores = np.asarray(stockfish_scores.iloc[i, 1].split(" "))
#             curr_scores = curr_scores.astype(np.int)
#             move_scores.append(curr_scores)
#
#     move_scores = np.asarray(move_scores)
#     print(move_scores)
#     return move_scores


def generate_training_data(score_dict, start, end, min_move_number, max_move_number):
    train_boards = []
    train_scores = []
    counter = 0
    for FEN in score_dict:
        if counter >= end:
            break

        if counter >= start:
            # Convert FEN string to one hot encoded chess board
            board = FEN_to_one_hot(FEN, min_move_number, max_move_number)

            # If white is winning, append 1 to labels. If black winning, append 0.
            # Here we ignore tied games because these will cause unnecessary complication.
            if board is not None:
                score = score_dict[FEN]
                if score >= 0:
                    train_scores.append(1)
                else:
                    train_scores.append(0)
                train_boards.append(board)

        counter += 1

    train_boards = np.asarray(train_boards)
    train_scores = np.asarray(train_scores)
    return train_boards, train_scores


def create_model():
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten(input_shape=(8, 8)))

    # Input tensor is 1D one hot encoded chess board
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu, input_dim=768))
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))

    # Last layer is probability of white/black advantage
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train(export_model, model_name, start, stop, test_set_size, epochs):
    # Get stockfish move evaluations for board states
    print('Importing FEN hash table...')
    score_dict = import_hash_table()
    print('Hash table size: ' + str(len(score_dict)))

    # Generate one hot encoded boards with their respective engine evaluations
    start_time = time.time()
    print('Generating training data...')
    train_boards, train_scores = generate_training_data(score_dict, start, stop)
    print('Training data finished (' + str(time.time() - start_time) + 's)')

    print(train_boards)
    print(train_scores)

    # Create test set by splitting part of training set
    test_boards = train_boards[len(train_boards) - test_set_size:]
    test_scores = train_scores[len(train_scores) - test_set_size:]

    # Remove test set from train set
    train_boards = train_boards[0:len(train_boards) - test_set_size:1]
    train_scores = train_scores[0:len(train_scores) - test_set_size:1]

    # Create model, and train
    model = create_model()
    print('Training...')
    model.fit(train_boards, train_scores, epochs=epochs)
    val_loss, val_acc = model.evaluate(test_boards, test_scores)
    print(val_loss)
    print(val_acc)

    # Save model
    if export_model:
        model.save(str(model_name))


def test_model(model_name, start, stop):
    if not str(model_name).endswith('.model'):
        model_name = str(model_name) + '.model'

    # Load model, and data from start to stop
    model = tf.keras.models.load_model(model_name)
    print('Importing FEN hash table...')
    score_dict = import_hash_table()
    print('Hash table size: ' + str(len(score_dict)))
    test_boards, test_scores = generate_training_data(score_dict, start, stop, 30, None)

    # Evaluate test data
    val_loss, val_acc = model.evaluate(test_boards, test_scores)
    print(val_loss)
    print(val_acc)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str2bool, nargs='?', help='(T/F) If training, or not.')
    parser.add_argument('--export_model', type=str2bool, nargs='?', help='(T/F) Export model after training')
    parser.add_argument('--model_name', default='chess_model.model',
                        help='Name of model to export/import (for training/testing).')
    parser.add_argument('--epochs', type=int, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, help='Training batch size.')
    parser.add_argument('--test_size', type=int, help='Size of data that is designated for testing when training.')

    parser.add_argument('--test', type=str2bool, nargs='?', help='(T/F) If testing, or not.')
    parser.add_argument('--start', type=int, help='Start reading data from this index.')
    parser.add_argument('--stop', type=int, help='Stop reading data from this index.')
    args = parser.parse_args()

    # Set Tf session to use GPU memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    if args.test:
        test_model(args.model_name, args.start, args.stop)
    elif args.train:
        train(args.export_model, args.model_name, args.start, args.stop, args.test_size, args.epochs)
    else:
        print('Invalid state. Either --train, or --test arguments must be true.')
    # main(True, True)
