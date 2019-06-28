import os

import chess
import numpy as np
import argparse
import time
from stockfish_uci import import_hash_table

# One hot encoded values for each piece:
empty = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
w_pawn = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
b_pawn = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
w_rook = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
b_rook = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
w_knight = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
b_knight = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
w_bishop = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
b_bishop = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
w_queen = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
b_queen = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
w_king = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
b_king = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

neither_attacking_tile = [0, 0]
white_attacking_tile = [1, 0]
black_attacking_tile = [0, 1]
both_attack_tile = [1, 1]

piece_dict = {'None': empty, 'P': w_pawn, 'p': b_pawn, 'R': w_rook,
              'r': b_rook, 'N': w_knight, 'n': b_knight, 'B': w_bishop,
              'b': b_bishop, 'Q': w_queen, 'q': b_queen, 'K': w_king,
              'k': b_king}


def FEN_to_one_hot(FEN, min_move_number=None, max_move_number=None):
    """
    Creates a one hot encoded flat array for a given chess
    board.
    :param FEN: The representation of chess board.
    :param min_move_number: Return None if board has not reached
    this move number.
    :param max_move_number: Return None if board has exceeded
     this move number.
    :return: a one hot encoded flat array for a given chess
    board.
    """
    # Default values for min/max move number if not given:
    if min_move_number is None:
        min_move_number = -1

    if max_move_number is None:
        max_move_number = 500

    board = chess.Board(FEN)
    white_turn = board.turn
    base_board = chess.BaseBoard(board.board_fen())
    if board.fullmove_number < min_move_number or board.fullmove_number > max_move_number:
        return None, None, None

    # Mirror board if blacks turn.
    # This avoids the problem of "white-to-play" vs. "black-to-play" when training.
    if not board.turn:
        board = board.mirror()
        base_board = base_board.mirror()

    # Create one hot encoded board:
    attack_matrix = np.zeros(64)
    pieces = []
    for pos in chess.SQUARES:
        white_attack = base_board.is_attacked_by(chess.WHITE, pos)
        black_attack = base_board.is_attacked_by(chess.BLACK, pos)

        # No attackers = 0, white attack = 1, black attack = 2, both = 3
        if white_attack:
            attack_matrix[chess.square_mirror(pos)] += 1
        if black_attack:
            attack_matrix[chess.square_mirror(pos)] += 2

        pieces.extend(piece_dict[str(board.piece_at(pos))])

    pieces = np.asarray(pieces)
    # pieces = np.reshape(pieces, (-1, 96))
    # attack_matrix = np.reshape(attack_matrix, (-1, 8))
    return pieces, attack_matrix, white_turn


def FEN_to_one_hot_no_attack_mat(FEN, min_move_number=None, max_move_number=None):
    """
    Creates a one hot encoded flat array for a given chess
    board.
    :param FEN: The representation of chess board.
    :param min_move_number: Return None if board has not reached
    this move number.
    :param max_move_number: Return None if board has exceeded
     this move number.
    :return: a one hot encoded flat array for a given chess
    board.
    """
    # Default values for min/max move number if not given:
    if min_move_number is None:
        min_move_number = -1

    if max_move_number is None:
        max_move_number = 500

    board = chess.Board(FEN)
    white_turn = board.turn
    base_board = chess.BaseBoard(board.board_fen())
    if board.fullmove_number < min_move_number or board.fullmove_number > max_move_number:
        return None, None, None

    # Mirror board if blacks turn.
    # This avoids the problem of "white-to-play" vs. "black-to-play" when training.
    if not board.turn:
        board = board.mirror()
        base_board = base_board.mirror()

    # Create one hot encoded board:
    pieces = []
    for pos in chess.SQUARES:
        pieces.extend(piece_dict[str(board.piece_at(pos))])

    pieces = np.asarray(pieces)
    # pieces = np.reshape(pieces, (-1, 96))
    # attack_matrix = np.reshape(attack_matrix, (-1, 8))
    return pieces, None, white_turn


def FEN_to_input(FEN, min_move_number=None, max_move_number=None):
    """
    Creates a one hot encoded flat array for a given chess
    board, and an attack matrix.
    :param FEN: The representation of chess board.
    :param min_move_number: Return None if board has not reached
    this move number.
    :param max_move_number: Return None if board has exceeded
     this move number.
    :return: a one hot encoded flat array for a given chess
    board.
    """
    # Default values for min/max move number if not given:
    if min_move_number is None:
        min_move_number = -1

    if max_move_number is None:
        max_move_number = 500

    board = chess.Board(FEN)
    white_turn = board.turn
    base_board = chess.BaseBoard(board.board_fen())
    if board.fullmove_number < min_move_number or board.fullmove_number > max_move_number:
        return None, None, None

    # Mirror board if blacks turn.
    # This avoids the problem of "white-to-play" vs. "black-to-play" when training.
    if not board.turn:
        board = board.mirror()
        base_board = base_board.mirror()

    # Create one hot encoded board and attack matrix
    attack_matrix = []
    pieces = []
    for pos in chess.SQUARES:
        white_attack = base_board.is_attacked_by(chess.WHITE, pos)
        black_attack = base_board.is_attacked_by(chess.BLACK, pos)

        # No attackers = 0, white attack = 1, black attack = 2, both = 3
        if white_attack and black_attack:
            # attack_matrix[chess.square_mirror(pos)] += 1
            attack_matrix.extend(both_attack_tile)
        elif white_attack:
            attack_matrix.extend(white_attacking_tile)
        elif black_attack:
            # attack_matrix[chess.square_mirror(pos)] += 2
            attack_matrix.extend(black_attacking_tile)
        else:
            attack_matrix.extend(neither_attacking_tile)

        pieces.extend(piece_dict[str(board.piece_at(pos))])

    pieces = np.asarray(pieces)
    # pieces = np.reshape(pieces, (-1, 96))
    # attack_matrix = np.reshape(attack_matrix, (-1, 8))
    return pieces, attack_matrix, white_turn


def parse_FEN_dict_to_file(dictionary):
    out_array = []
    start = time.time()
    counter = 0
    for FEN in dictionary:
        if counter > 10000:
            break
        board, attack_matrix, turn = FEN_to_one_hot(FEN, None, None)
        score = dictionary[FEN]
        entry = np.asarray([board, attack_matrix, turn, score])
        out_array.append(entry)
        counter += 1

    out_array = np.asarray(out_array)
    print('Export complete. Entries: ' + str(len(out_array)))
    print('Elapsed time in seconds: ' + str(time.time() - start))
    np.save('training_data_' + str(len(out_array)), out_array)


def import_hash_table(file):
    score_dict = {}
    # Attempt to load already processed boards
    if os.path.isfile(file):
        score_dict = np.load(file).item()
    else:
        print('No hash table found.')

    return score_dict


if __name__ == '__main__':
    board, attack_matrix, turn = (FEN_to_input('3r1rk1/3nbppp/p2q1n2/1p6/3Pp3/1QN1P3/PP1BBPPP/2R2RK1 w - - 6 17'))

    index = 0
    for val in attack_matrix:
        if index == 23:
            print(val)
            index = 0
        else:
            print(val, end=" ")
        index += 1
    # test_dict = import_hash_table()
    # parse_FEN_dict_to_file(test_dict)
    # x = np.load('training_data_10001.npy')
    # print(x)
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--fen', help='FEN board representation.')
    # args = parser.parse_args()
