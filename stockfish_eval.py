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

        self.import_hash_table('threads\\FICS\\known_scores_merged.npy')

    def import_hash_table(self, dict_file):
        # Attempt to load already processed boards
        if os.path.isfile(dict_file):
            self.score_dict = np.load(dict_file).item()
            print('Imported hash table of length {}', str(len(self.score_dict)))
        else:
            print('No hash table found. Creating new hash table.')

    def export_hash_table(self):
        np.save(self.score_dict_filename, self.score_dict)

    def eval_thread(self, thread_num, pgn_file=None):
        engine = chess.uci.popen_engine(self.stockfish_exe)
        if pgn_file is None:
            pgn = open(self.pgn_file)
        else:
            pgn = open(pgn_file)

        def export_thread_hash_table():
            print('Saving progress for thread {} len: {}'.format(thread_num, len(thread_score_dict)))
            filename = str(self.score_dict_filename) + '_' + str(thread_num) + '.npy'
            np.save('threads\\' + filename, thread_score_dict)

        engine.uci()
        engine.setoption({"Threads": 1, "Hash": 64})
        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)

        game_num = 0
        games_processed_by_thread = 0

        if pgn_file is None:
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
                if board.fen() not in self.score_dict and \
                        board.fen() not in thread_score_dict:
                    engine.position(board)

                    # Provide deeper evaluation for earlier moves
                    # if move_number < 5:
                    #     eval_time = 300
                    # elif move_number < 15:
                    #     eval_time = 200
                    # else:
                    #     eval_time = 200

                    try:
                        engine.go(movetime=200, ponder=False)
                    except chess.uci.EngineTerminatedException as err:
                        print('Unexpected engine error:')
                        print(err)

                    engine.stop()
                    # engine.go(depth=25, ponder=False)
                    print(info_handler.info['score'][1][0])
                    print('----')
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
            if pgn_file is None:
                skip_to = self.threads + game_num
                while game_num < skip_to:
                    chess.pgn.skip_game(pgn)
                    game_num += 1
            else:
                game_num += 1

            game = chess.pgn.read_game(pgn)
            games_processed_by_thread += 1
            if games_processed_by_thread % self.export_inc == 0:
                export_thread_hash_table()

    def execute_parallel_eval(self):
        procs = []
        # If pgn_file is a list of pgn's then assign them to threads.
        if type(self.pgn_file) is list:
            for i in range(self.threads):
                thread_pgn = self.pgn_file[i]
                print('Thread ' + str(i) + ' started. PGN: ' + str(thread_pgn))
                p = threading.Thread(target=self.eval_thread, args=(i, thread_pgn))
                procs.append(p)
                p.start()
        else:
            for i in range(self.threads):
                print('Thread ' + str(i) + ' started.')
                # p = Process(target=self.eval_thread, args=(i, ))
                p = threading.Thread(target=self.eval_thread, args=(i,))
                procs.append(p)
                p.start()

        for proc in procs:
            proc.join()


if __name__ == '__main__':
    # KingBase_pgns = ['data\\KingBase\\KingBase2019-A00-A39.pgn',
    #                  'data\\KingBase\\KingBase2019-A40-A79.pgn',
    #                  'data\\KingBase\\KingBase2019-A80-A99.pgn',
    #                  'data\\KingBase\\KingBase2019-B00-B19.pgn',
    #                  'data\\KingBase\\KingBase2019-B20-B49.pgn',
    #                  'data\\KingBase\\KingBase2019-B50-B99.pgn',
    #                  'data\\KingBase\\KingBase2019-C00-C19.pgn',
    #                  'data\\KingBase\\KingBase2019-C20-C59.pgn',
    #                  'data\\KingBase\\KingBase2019-C60-C99.pgn',
    #                  'data\\KingBase\\KingBase2019-D00-D29.pgn',
    #                  'data\\KingBase\\KingBase2019-D30-D69.pgn',
    #                  'data\\KingBase\\KingBase2019-D70-D99.pgn',
    #                  'data\\KingBase\\KingBase2019-E00-E19.pgn',
    #                  'data\\KingBase\\KingBase2019-E20-E59.pgn',
    #                  'data\\KingBase\\KingBase2019-E60-E99.pgn']

    # ficsgamesdb_2018_chess2000_nomovetimes_79725
    # ficsgamesdb_2017_chess2000_nomovetimes_80572
    evaluator = StockfishEval('Stockfish\\stockfish_10_x64.exe',
                              'data\\FICS\\ficsgamesdb_2017_chess2000_nomovetimes_80572.pgn',
                              'TEST',
                              16,
                              75,)

    evaluator.execute_parallel_eval()
