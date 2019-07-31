import chess
import numpy as np
import time
import os
import random


class TrainingDataGenerator:
    """
    Generates MLP/CNN input data from a dictionary of {FEN String:CP Score}
    """
    def __init__(self,
                 score_dict_file,
                 num_classes=2,
                 flatten=True,
                 min_move_num=0,
                 max_move_num=500):

        np.random.seed(42)
        if not str(score_dict_file).endswith('.npy'):
            score_dict_file += '.npy'

        self.score_dict = self.import_score_dict(score_dict_file)
        self.num_classes = num_classes
        self.flatten = flatten
        self.min_move_num = min_move_num
        self.max_move_num = max_move_num
        self.SquareSet = chess.SquareSet(
            chess.BB_A1 | chess.BB_A2 | chess.BB_A3 | chess.BB_A4 | chess.BB_A5 |
            chess.BB_A6 | chess.BB_A7 | chess.BB_A8 |
            chess.BB_B1 | chess.BB_B2 | chess.BB_B3 | chess.BB_B4 | chess.BB_B5 |
            chess.BB_B6 | chess.BB_B7 | chess.BB_B8 |
            chess.BB_C1 | chess.BB_C2 | chess.BB_C3 | chess.BB_C4 | chess.BB_C5 |
            chess.BB_C6 | chess.BB_C7 | chess.BB_C8 |
            chess.BB_D1 | chess.BB_D2 | chess.BB_D3 | chess.BB_D4 | chess.BB_D5 |
            chess.BB_D6 | chess.BB_D7 | chess.BB_D8 |
            chess.BB_E1 | chess.BB_E2 | chess.BB_E3 | chess.BB_E4 | chess.BB_E5 |
            chess.BB_E6 | chess.BB_E7 | chess.BB_E8 |
            chess.BB_F1 | chess.BB_F2 | chess.BB_F3 | chess.BB_F4 | chess.BB_F5 |
            chess.BB_F6 | chess.BB_F7 | chess.BB_F8 |
            chess.BB_G1 | chess.BB_G2 | chess.BB_G3 | chess.BB_G4 | chess.BB_G5 |
            chess.BB_G6 | chess.BB_G7 | chess.BB_G8 |
            chess.BB_H1 | chess.BB_H2 | chess.BB_H3 | chess.BB_H4 | chess.BB_H5 |
            chess.BB_H6 | chess.BB_H7 | chess.BB_H8
        )
        self.ImportantSquareSet = chess.SquareSet(
            chess.BB_D4 | chess.BB_D5 |
            chess.BB_C4 | chess.BB_C5 |
            chess.BB_E4 | chess.BB_E5 |
            chess.BB_F2 | chess.BB_F7 |
            chess.BB_H2 | chess.BB_H7
        )

    def import_score_dict(self, file):
        score_dict = {}
        # Attempt to load already processed boards
        if os.path.isfile(file):
            score_dict = np.load(file).item()
        else:
            print('No hash table found.')

        return score_dict

    def parse_FEN_dict(self, num_boards=-1):
        boards = []
        scores = []
        start = time.time()
        print('Generating training data for ' + str(len(self.score_dict)) + " boards.")
        items = list(self.score_dict.items())
        random.shuffle(items)

        counter = 0
        # MLP:
        if self.flatten:
            for FEN, score in items:
                if 0 < num_boards < counter:
                    break

                board, parsed_score, turn = self.parse_FEN(FEN, score)
                if board is not None and parsed_score is not None and turn is not None:
                    boards.append(board)
                    if self.num_classes == 2:
                        scores.append(parsed_score)
                    else:
                        scores.append(self.score_to_ternary(score))

                counter += 1
        # CNN:
        else:
            for FEN, score in items:
                if 0 < num_boards < counter:
                    break

                board, parsed_score, turn = self.parse_FEN_3D(FEN, score)
                if board is not None and score is not None:
                    boards.append(board)
                    if self.num_classes == 2:
                        scores.append(parsed_score)
                    else:
                        scores.append(self.score_to_ternary(score))

                counter += 1

        end = time.time()
        boards = np.asarray(boards)
        scores = np.asarray(scores)
        print('Elapsed time: ' + str(end - start))

        return boards, scores

    # FEN to MLP input
    def parse_FEN(self, FEN, score):
        board = chess.Board(FEN)
        turn = board.turn

        if board.fullmove_number < self.min_move_num \
                or board.fullmove_number > self.max_move_num:
            return None, None, None

        # Mirror board on blacks turn, and negate score
        # if not board.turn:
        #     board = board.mirror()
        #     score = score * -1

        if score >= 0:
            binary_score = 1
        else:
            binary_score = 0

        w_pawn = np.asarray(board.pieces(chess.PAWN, chess.WHITE).tolist()).astype(int)
        w_rook = np.asarray(board.pieces(chess.ROOK, chess.WHITE).tolist()).astype(int)
        w_knight = np.asarray(board.pieces(chess.KNIGHT, chess.WHITE).tolist()).astype(int)
        w_bishop = np.asarray(board.pieces(chess.BISHOP, chess.WHITE).tolist()).astype(int)
        w_queen = np.asarray(board.pieces(chess.QUEEN, chess.WHITE).tolist()).astype(int)
        w_king = np.asarray(board.pieces(chess.KING, chess.WHITE).tolist()).astype(int)

        b_pawn = (np.asarray(board.pieces(chess.PAWN, chess.BLACK).tolist()) * -1).astype(int)
        b_rook = (np.asarray(board.pieces(chess.ROOK, chess.BLACK).tolist()) * -1).astype(int)
        b_knight = (np.asarray(board.pieces(chess.KNIGHT, chess.BLACK).tolist()) * -1).astype(int)
        b_bishop = (np.asarray(board.pieces(chess.BISHOP, chess.BLACK).tolist()) * -1).astype(int)
        b_queen = (np.asarray(board.pieces(chess.QUEEN, chess.BLACK).tolist()) * -1).astype(int)
        b_king = (np.asarray(board.pieces(chess.KING, chess.BLACK).tolist()) * -1).astype(int)

        # White/Black check, or no check
        if board.is_check() and board.turn is True:
            white_checked = 1
            black_checked = 0
        elif board.is_check() and board.turn is False:
            white_checked = 0
            black_checked = 1
        else:
            white_checked = 0
            black_checked = 0

        # [turn, white check, black check] bits
        turn_check_bits = np.asarray([turn, white_checked, black_checked], dtype=int)

        binary_board = np.concatenate((w_pawn, w_rook, w_knight, w_bishop, w_queen, w_king,
                                       b_pawn, b_rook, b_knight, b_bishop, b_queen, b_king,
                                       turn_check_bits))

        return binary_board, binary_score, turn

    # FEN to CNN input
    def parse_FEN_3D(self, FEN, score):
        board = chess.Board(FEN)
        turn = board.turn

        if board.fullmove_number < self.min_move_num \
                or board.fullmove_number > self.max_move_num:
            return None, None

        # Mirror board on blacks turn, and negate score
        # if not board.turn:
        # board = board.mirror()
        # score = score * -1

        if score >= 0:
            score = 1
        else:
            score = 0

        w_pawn = np.reshape(board.pieces(chess.PAWN, chess.WHITE).tolist(), (-1, 8)).astype(int)
        w_rook = np.reshape(board.pieces(chess.ROOK, chess.WHITE).tolist(), (-1, 8)).astype(int)
        w_knight = np.reshape(board.pieces(chess.KNIGHT, chess.WHITE).tolist(), (-1, 8)).astype(int)
        w_bishop = np.reshape(board.pieces(chess.BISHOP, chess.WHITE).tolist(), (-1, 8)).astype(int)
        w_queen = np.reshape(board.pieces(chess.QUEEN, chess.WHITE).tolist(), (-1, 8)).astype(int)
        w_king = np.reshape(board.pieces(chess.KING, chess.WHITE).tolist(), (-1, 8)).astype(int)

        b_pawn = (np.reshape(board.pieces(chess.PAWN, chess.BLACK).tolist(), (-1, 8)) * -1).astype(int)
        b_rook = (np.reshape(board.pieces(chess.ROOK, chess.BLACK).tolist(), (-1, 8)) * -1).astype(int)
        b_knight = (np.reshape(board.pieces(chess.KNIGHT, chess.BLACK).tolist(), (-1, 8)) * -1).astype(int)
        b_bishop = (np.reshape(board.pieces(chess.BISHOP, chess.BLACK).tolist(), (-1, 8)) * -1).astype(int)
        b_queen = (np.reshape(board.pieces(chess.QUEEN, chess.BLACK).tolist(), (-1, 8)) * -1).astype(int)
        b_king = (np.reshape(board.pieces(chess.KING, chess.BLACK).tolist(), (-1, 8)) * -1).astype(int)

        checked_info = []

        if board.turn is True:
            turn = [1] * 64
        else:
            turn = [0] * 64

        if board.is_check() and board.turn is True:
            checked_info = [-1] * 64

        elif board.is_check() and board.turn is False:
            checked_info = [1] * 64

        elif not board.is_check():
            checked_info = [0] * 64

        square_attackers = []
        pinned_squares = []
        important_attackers_features = []

        for square in self.SquareSet:
            if board.is_attacked_by(chess.WHITE, square):
                square_attackers.append(1)
            elif board.is_attacked_by(chess.BLACK, square):
                square_attackers.append(-1)
            else:
                square_attackers.append(0)

            if board.is_pinned(chess.WHITE, square):
                pinned_squares.append(1)
            elif board.is_pinned(chess.BLACK, square):
                pinned_squares.append(-1)
            else:
                pinned_squares.append(0)

        for ImportantSquare in self.ImportantSquareSet:
            WhiteAttackers = board.attackers(chess.WHITE, ImportantSquare)
            BlackAttackers = board.attackers(chess.BLACK, ImportantSquare)

            if len(WhiteAttackers) > len(BlackAttackers):
                important_attackers_features = [1] * 64
            elif len(WhiteAttackers) < len(BlackAttackers):
                important_attackers_features = [-1] * 64
            else:
                important_attackers_features = [0] * 64

        turn = np.asarray(turn)
        checked_info = np.asarray(checked_info)
        square_attackers = np.asarray(square_attackers)
        pinned_squares = np.asarray(pinned_squares)
        important_attackers_features = np.asarray(important_attackers_features)

        turn = np.reshape(turn, (-1, 8))
        checked_info = np.reshape(checked_info, (-1, 8))
        square_attackers = np.reshape(square_attackers, (-1, 8))
        pinned_squares = np.reshape(pinned_squares, (-1, 8))
        important_attackers_features = np.reshape(important_attackers_features, (-1, 8))
        binary_board = np.dstack((w_pawn, w_rook, w_knight, w_bishop, w_queen, w_king,
                                  b_pawn, b_rook, b_knight, b_bishop, b_queen, b_king,
                                  turn, checked_info, square_attackers, pinned_squares, important_attackers_features))

        # print(checked_info)
        # print(square_attackers)
        # print(pinned_squares)
        # print(important_attackers_features)

        return binary_board, score, turn

    # Label score
    def score_to_ternary(self, score):
        if -150 < score <= 150:
            return np.asarray([1, 0, 0])
        elif score > 150:
            return np.asarray([0, 1, 0])
        elif score <= -150:
            return np.asarray([0, 0, 1])
