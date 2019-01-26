import chess
import numpy as np
import argparse

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

piece_dict = {'None': empty, 'P': w_pawn, 'p': b_pawn, 'R': w_rook,
              'r': b_rook, 'N': w_knight, 'n': b_knight, 'B': w_bishop,
              'b': b_bishop, 'Q': w_queen, 'q': b_queen, 'K': w_king,
              'k': b_king}


def FEN_to_one_hot(FEN, min_move_number = None, max_move_number = None):
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
    if board.fullmove_number < min_move_number or board.fullmove_number > max_move_number:
        return None

    # Mirror board if blacks turn.
    # This avoids the problem of "white-to-play" vs
    # "black-to-play" when training.
    if not board.turn:
        board = board.mirror()

    # Create one hot encoded board:
    pieces = []
    for pos in chess.SQUARES:
        pieces.extend(piece_dict[str(board.piece_at(pos))])

    pieces = np.asarray(pieces)
    return pieces


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fen', help='FEN board representation.')
    args = parser.parse_args()

    print(FEN_to_one_hot(str(args.fen)))
