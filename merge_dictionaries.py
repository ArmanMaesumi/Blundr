import numpy as np
import os
import time


def merge_dicts(dict_paths):
    """
    Merges all dictionary files in list of dictionary paths
    :param dict_paths: List of dictionary paths
    :return: A union of all dictionaries
    """
    merged = {}
    for dict_path in dict_paths:
        dictionary = np.load(dict_path).item()
        merged = {**merged, **dictionary}

    return merged


def merge_all_in_path():
    """
    Merges all dictionaries in the script path.
    Exports the merged dictionary as 'known_scores_merged.npy'
    """
    start = time.time()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(script_dir)
    dict_dirs = []
    for file in os.listdir(script_dir):
        if file.endswith('.npy'):
            dict_dirs.append(os.path.join(script_dir, file))

    print(str(len(dict_dirs)) + ' Dictionaries found. Merging...')
    print(dict_dirs)
    merged_dictionary = merge_dicts(dict_dirs)
    print('Dictionaries merged. Merged length: ' + str(len(merged_dictionary)))
    print('Merge elapsed time: ' + str((time.time() - start)) + 's')
    print('Exporting dictionary...')
    start = time.time()
    export_merged_dict(merged_dictionary)
    print('Merged dictionary exported successfully. Export elapsed time: ' + str((time.time() - start)) + 's')


def export_merged_dict(dictionary):
    np.save('known_scores_merged.npy', dictionary)


if __name__ == '__main__':
    merge_all_in_path()
