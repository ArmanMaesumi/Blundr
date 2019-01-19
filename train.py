import tensorflow as tf
import chess.pgn
import numpy as np
import pandas
import time

empty = 0
w_pawn = 1
b_pawn = 2
w_rook = 3
b_rook = 4
w_knight = 5
b_knight = 6
w_bishop = 7
b_bishop = 8
w_queen = 9
b_queen = 10
w_king = 11
b_king = 12

piece_dict = {'.': empty, 'P': w_pawn, 'p': b_pawn, 'R': w_rook,
              'r': b_rook, 'N': w_knight, 'n': b_knight, 'B': w_bishop,
              'b': b_bishop, 'Q': w_queen, 'q': b_queen, 'K': w_king,
              'k': b_king}


# Create chess board from FEN string
def board_arr_from_fen(FEN):
    board = chess.Board(FEN)
    return board_to_array(board)


# Convert chess board to array using piece_dict values
def board_to_array(board):
    board_arr = []
    board_str_arr = str(board).split("\n")
    for row in board_str_arr:
        curr_row = row.split(" ")
        board_arr.append(list(map(piece_dict.get, curr_row)))

    return board_arr


def parse_stockfish_scores(num_scores):
    stockfish_csv = 'data\\stockfish_scores.csv'
    names = ['Event', 'MoveScores']
    stockfish_scores = pandas.read_csv(stockfish_csv, names=names)
    move_scores = []

    # Read in stockfish evaluation for player moves
    for i in range(num_scores):
        if 0 < i:
            curr_scores = np.asarray(stockfish_scores.iloc[i, 1].split(" "))
            curr_scores = curr_scores.astype(np.int)
            move_scores.append(curr_scores)

    move_scores = np.asarray(move_scores)
    print(move_scores)
    return move_scores


def generate_training_data(move_scores, white_to_play):
    pgn = open('data\\data.pgn')

    train_boards = []
    train_scores = []

    # Iterate over games in pgn
    for game_num in range(len(move_scores)):
        game = chess.pgn.read_game(pgn)
        board = game.board()
        move_num = 0

        # Iterate over moves in current game
        for move in game.mainline_moves():
            board.push(move)
            # Gather white-to-play or black-to-play positions depending on which model we are training
            if ((move_num % 2) == 0 and white_to_play) or ((move_num % 2) != 0 and not white_to_play):
                board_arr = board_to_array(board)
                train_boards.append(board_arr)
                score = move_scores[game_num][move_num]
                # Score >= 0 means white advantage (0), else black advantage (1)
                if score >= 0:
                    train_scores.append(0)
                else:
                    train_scores.append(1)
            move_num += 1

    train_boards = np.asarray(train_boards)
    train_scores = np.asarray(train_scores)
    return train_boards, train_scores


def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

    # Last layer is probability of white/black advantage
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def main(white_to_play, export_model):
    # Set Tf session to use GPU memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    # Get stockfish move evaluations
    start = time.time()
    print('Generating training data...')
    move_scores = parse_stockfish_scores(45000)
    train_boards, train_scores = generate_training_data(move_scores, True)
    end = time.time()
    print('Training data finished (' + str(end - start) + 's)')

    # Create test set by splitting part of training set
    test_set_size = 1000
    test_boards = train_boards[len(train_boards) - test_set_size:]
    test_scores = train_scores[len(train_scores) - test_set_size:]

    # Remove test set from train set
    train_boards = train_boards[0:len(train_boards) - test_set_size:1]
    train_scores = train_scores[0:len(train_scores) - test_set_size:1]

    # Create model, and train
    model = create_model()
    print('Training...')
    model.fit(train_boards, train_scores, epochs=3)
    val_loss, val_acc = model.evaluate(test_boards, test_scores)
    print(val_loss)
    print(val_acc)

    # Save model
    if export_model:
        model_name = 'chess_model_white.model' if white_to_play else 'chess_model_black.model'
        model.save(model_name)

if __name__ == "__main__":
    main(True, False)
