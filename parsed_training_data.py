import numpy as np


def export_nparray(np_array, file):
    np.save(file, np_array)


def import_nparray(file):
    return np.load(file)


def export_parsed_training_data(boards, attack_mats, scores, filename='parsed_training_data.npy'):

    if not filename.endswith('.npy'):
        filename = filename + '.npy'

    entries = []

    for i in range(len(boards)):
        entries.append(np.asarray([boards[i], attack_mats[i], scores[i]]))

    entries = np.asarray(entries)
    export_nparray(entries, filename)


def import_parsed_training_data(file):
    entries = import_nparray(file)
    print(entries[0])
    boards = []
    attack_mats = []
    scores = []

    for entry in entries:
        boards.append(entry[0])
        attack_mats.append(entry[1])
        scores.append(entry[2])

    boards = np.asarray(boards)
    attack_mats = np.asarray(attack_mats)
    scores = np.asarray(scores)
    print('------------------')
    return boards, attack_mats, scores


# if __name__ == '__main__':
#     test_board = np.zeros(786)
#     test_attack_mat = np.zeros(64)
#     test_score = np.asarray([1])
#     entries = []
#
#     start = time.time()
#     for i in range(2700000):
#         entries.append(np.asarray([test_board, test_attack_mat, test_score]))
#     entries = np.asarray(entries)
#     # print(entries)
#     export_nparray(entries, 'test_np_export.npy')
#     print('Time to export: ' + str(time.time() - start))
#     x = import_nparray('test_np_export.npy')
#     print(x)
#     print('done')
