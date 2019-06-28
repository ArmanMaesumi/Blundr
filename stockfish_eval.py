import os
import chess, chess.uci, chess.pgn
import numpy as np
import threading


class StockfishEval:
    """
    Evaluates board positions from a .pgn file in parallel.
    Saves a npy dictionary of FEN positions and their Stockfish evaluations.
    """
    def __init__(self,
                 stockfish_exe,
                 pgn_file,
                 score_dict_filename,
                 threads,
                 export_inc):
        if not str(score_dict_filename).endswith('.npy'):
            score_dict_filename += '.npy'

        self.stockfish_exe = stockfish_exe
        self.pgn_file = pgn_file
        self.score_dict_filename = score_dict_filename
        self.threads = int(threads)
        self.export_inc = export_inc
        self.score_dict = {}

        self.pgn = open(self.pgn_file)
        self.import_hash_table()

    def import_hash_table(self):
        # Attempt to load already processed boards
        if os.path.isfile(self.score_dict_filename):
            self.score_dict = np.load(self.score_dict_filename).item()
            print('Imported hash table of length {}', str(len(self.score_dict)))
        else:
            print('No hash table found. Creating new hash table.')

    def export_hash_table(self):
        np.save(self.score_dict_filename, self.score_dict)

    def eval_thread(self, thread_num):
        engine = chess.uci.popen_engine(self.stockfish_exe)
        pgn = open(self.pgn_file)

        def export_thread_hash_table():
            print('Saving progress for thread {} len: {}'.format(thread_num, len(thread_score_dict)))
            filename = str(self.score_dict_filename) + '{}.npy'.format(thread_num)
            np.save('threads\\' + filename, thread_score_dict)

        engine.uci()
        engine.setoption({"Threads": 1, "Hash": 64})
        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)

        game_num = 0
        games_processed_by_thread = 0
        while game_num < thread_num:
            chess.pgn.skip_game(pgn)
            game_num += 1

        game = chess.pgn.read_game(pgn)
        thread_score_dict = {}
        while game is not None:
            board = game.board()
            engine.ucinewgame()
            print('Processing game {} on thread {}'.format(game_num, thread_num))

            move_number = 0
            for move in game.mainline_moves():
                board.push(move)

                # Check if board has already been evaluated
                if board.fen() not in self.score_dict:
                    engine.position(board)

                    # Provide deeper evaluation for earlier moves
                    if move_number < 5:
                        eval_time = 300
                    elif move_number < 15:
                        eval_time = 200
                    else:
                        eval_time = 200

                    try:
                        engine.go(movetime=eval_time, ponder=False)
                    except chess.uci.EngineTerminatedException as err:
                        print('Unexpected engine error:')
                        print(err)

                    engine.stop()
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

                    thread_score_dict[board.fen()] = score

                move_number += 1

            # game = chess.pgn.read_game(self.pgn)
            skip_to = self.threads + game_num
            while game_num < skip_to:
                chess.pgn.skip_game(pgn)
                game_num += 1
            game = chess.pgn.read_game(pgn)
            games_processed_by_thread += 1
            if games_processed_by_thread % self.export_inc == 0:
                export_thread_hash_table()

    def execute_parallel_eval(self):
        procs = []
        for i in range(self.threads):
            print('Thread ' + str(i) + ' started.')
            # p = Process(target=self.eval_thread, args=(i, ))
            p = threading.Thread(target=self.eval_thread, args=(i,))
            procs.append(p)
            p.start()

        for proc in procs:
            proc.join()


if __name__ == '__main__':
    evaluator = StockfishEval('Stockfish\\stockfish_10_x64.exe',
                              'data\\lichess_db_standard_rated_2016-07.pgn',
                              'parallel_test',
                              16,
                              25)

    evaluator.execute_parallel_eval()
