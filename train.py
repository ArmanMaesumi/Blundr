import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.models import Model

from parse_FEN import FEN_to_one_hot
from stockfish_uci import import_hash_table
from parsed_training_data import export_parsed_training_data, import_parsed_training_data


def generate_training_data(score_dict, start, end, min_move_number, max_move_number):
    train_boards = []
    train_attack_matrix = []
    train_scores = []
    counter = 0
    for FEN in score_dict:
        if counter >= end:
            break

        if counter >= start:
            # Convert FEN string to one hot encoded chess board
            board, attack_matrix, turn = FEN_to_one_hot(FEN, min_move_number, max_move_number)

            # If white is winning, append 1 to labels. If black winning, append 0.
            # Here we ignore tied games because these will cause unnecessary complication.
            if board is not None and attack_matrix is not None:
                score = score_dict[FEN]
                if turn:
                    if score >= 0:
                        train_scores.append(1)
                    else:
                        train_scores.append(0)
                else:
                    if score >= 0:
                        train_scores.append(0)
                    else:
                        train_scores.append(1)

                train_boards.append(board)
                train_attack_matrix.append(attack_matrix)

        counter += 1

    train_boards = np.asarray(train_boards)
    train_attack_matrix = np.asarray(train_attack_matrix)
    train_scores = np.asarray(train_scores)
    return train_boards, train_attack_matrix, train_scores


def create_model():
    # TODO: make attack mat 1d one-hot encoded?
    board_input = Input(shape=(768,), name='board_input')
    x = Dense(1024, activation='relu')(board_input)

    attack_matrix = Input(shape=(64,), name='attack_matrix')
    y = Dense(1024, activation='relu')(attack_matrix)

    x = tf.keras.layers.concatenate([x, y])
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[board_input, attack_matrix], outputs=[main_output])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train(export_model, model_name, start, stop, test_set_size, epochs, export_training_data, import_training_data):
    # Get stockfish move evaluations for board states
    print('Importing FEN hash table...')
    score_dict = import_hash_table()
    print('Hash table size: ' + str(len(score_dict)))

    # Generate one hot encoded boards with their respective engine evaluations
    start_time = time.time()
    print('Generating training data...')

    # Instead of parsing training data, try to import pre-parsed data:
    if import_training_data is None:
        train_boards, train_attack_matrix, train_scores = generate_training_data(score_dict, start, stop, None, None)
        train_attack_matrix = tf.keras.utils.normalize(train_attack_matrix)
    else:
        train_boards, train_attack_matrix, train_scores = import_parsed_training_data(str(import_training_data))

    if export_training_data is not None:
        export_parsed_training_data(train_boards, train_attack_matrix, train_scores, str(export_training_data))

    print('Training data finished (' + str(time.time() - start_time) + 's)')

    # # Create test set by splitting part of training set
    # test_boards = train_boards[len(train_boards) - test_set_size:]
    # test_scores = train_scores[len(train_scores) - test_set_size:]
    #
    # # Remove test set from train set
    # train_boards = train_boards[0:len(train_boards) - test_set_size:1]
    # train_scores = train_scores[0:len(train_scores) - test_set_size:1]

    # Create model, and train
    model = create_model()
    print('Training...')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    model.fit([train_boards, train_attack_matrix], [train_scores],
              epochs=epochs, validation_split=0.05, callbacks=[tensorboard])

    # val_loss, val_acc = model.evaluate(test_boards, test_scores)
    # print(val_loss)
    # print(val_acc)

    # Save model
    if export_model:
        model.save(str(model_name))


def test_model(model_name, start, stop):
    # --test T --export_model T --model_name chess_model_Conv2D.model --epochs 10 --test_size 100 --start 2200000 --stop 2250000
    if not str(model_name).endswith('.model'):
        model_name = str(model_name) + '.model'

    # Load model, and data from start to stop
    model = tf.keras.models.load_model(model_name)
    print('Importing FEN hash table...')
    score_dict = import_hash_table()
    print('Hash table size: ' + str(len(score_dict)))
    test_boards, test_attack_matrix, test_scores = generate_training_data(score_dict, start, stop, 4, None)
    test_attack_matrix = tf.keras.utils.normalize(test_attack_matrix)
    # Evaluate test data
    val_loss, val_acc = model.evaluate([test_boards, test_attack_matrix], [test_scores])
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
    # --train T --export_model T --model_name chess_model_attack_matv4.model --epochs 50 --test_size 50000 --start 1 --stop 2600000 --import_parsed_training_data parsed_training_data.npy
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str2bool, nargs='?', help='(T/F) If training, or not.')
    parser.add_argument('--export_model', type=str2bool, nargs='?', help='(T/F) Export model after training')
    parser.add_argument('--export_parsed_training_data', help='Export parsed training data.')
    parser.add_argument('--import_parsed_training_data', help='Import parsed training data.')
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
        train(args.export_model, args.model_name, args.start, args.stop, args.test_size, args.epochs,
              args.export_parsed_training_data, args.import_parsed_training_data)
    else:
        print('Invalid state. Either --train, or --test arguments must be true.')
        # main(True, True)
