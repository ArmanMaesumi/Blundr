import chess.uci
import chess.pgn
import numpy as np
import csv

# Pre-load every white opening
past_board_scores = {'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1': 10,
                     'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1': 10,
                     'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1': 10,
                     'rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq - 0 1': -10,
                     'rnbqkbnr/pppppppp/8/8/6P1/8/PPPPPP1P/RNBQKBNR b KQkq - 0 1': -80,
                     'rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1': -10,
                     'rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq - 0 1': -30,
                     'rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq - 0 1': -60,
                     'rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1': -10,
                     'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1': 10}

engine = chess.uci.popen_engine("stockfish\\stockfish_10_x64.exe")
engine.uci()
engine.setoption({"Threads": 10, "Hash": 64})
pgn = open("data\\lichess_db_standard_rated_2013-03.pgn")


def evaluate_pgn_moves():
    info_handler = chess.uci.InfoHandler()
    engine.info_handlers.append(info_handler)

    # test_board = chess.Board('8/7Q/8/1R4pk/6p1/4P1P1/1P3PK1/8 b - - 6 49')
    # engine.position(test_board)
    # print(engine.go(movetime=1000))
    # print(info_handler.info['score'][1])

    scores = []
    game_num = 1
    game = chess.pgn.read_game(pgn)
    csv_fields = ['Event', 'MoveScores']
    with open(r'lichess_db_standard_rated_2013-03.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_fields)

    while game is not None and game_num < 100000:
        board = game.board()
        match_scores = []
        engine.ucinewgame()
        move_number = 0
        for move in game.mainline_moves():
            board.push(move)

            # Check if board has already been evaluated
            if board.fen() not in past_board_scores:
                engine.position(board)

                # Provide deeper evaluation for earlier moves
                if move_number < 5:
                    eval_time = 3000
                elif move_number < 15:
                    eval_time = 2000
                else:
                    eval_time = 1000

                engine.go(movetime=eval_time, ponder=False)
                # engine.go(depth=25, ponder=False)
                score = info_handler.info['score'][1].cp
                mate = info_handler.info['score'][1].mate

                # If Stockfish finds mate, then give an extreme score
                if mate is not None:
                    if mate > 0:
                        if board.turn:
                            score = 10000
                        else:
                            score = -10000
                    else:
                        if board.turn:
                            score = -10000
                        else:
                            score = 10000
                elif not board.turn:
                    # Adjust score if Stockfish is playing black's turn
                    score *= -1

                match_scores.append(score)
                past_board_scores[board.fen()] = score
            else:
                known_score = past_board_scores[board.fen()]
                match_scores.append(known_score)

            move_number += 1
            print(match_scores)

        game = chess.pgn.read_game(pgn)
        scores.append(match_scores)
        print(scores)
        with open('lichess_db_standard_rated_2013-03.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            entry = [game_num, str(match_scores).replace(',', '').replace('[', '').replace(']', '')]
            writer.writerow(entry)
        export_hash_table()
        game_num += 1


def export_hash_table():
    # Save dict of already processed boards
    np.save('known_scores.npy', past_board_scores)
    print(past_board_scores)


def import_hash_table():
    global past_board_scores
    past_board_scores = np.load('known_scores.npy').item()
    print(past_board_scores)


def main():
    import_hash_table()
    evaluate_pgn_moves()


if __name__ == '__main__':
    main()



