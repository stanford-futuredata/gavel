import argparse
import numpy as np
import pickle
import random

from fancyimpute import NuclearNormMinimization

def mse(matrix, matrix_with_missing_entries, missing_entries):
    predicted_matrix = NuclearNormMinimization(
        verbose=False).fit_transform(matrix_with_missing_entries)
    error = []
    for (i, j) in missing_entries:
        error.append(matrix[(i, j)] - predicted_matrix[(i, j)])
    return (np.array(error) ** 2).mean()

def matrix_completion(matrices_filename, p_drops, num_trials):
    matrices = pickle.load(open(matrices_filename, 'rb'))
    for gpu_architecture in matrices:
        print "===================================================================="
        print "                              %s" % gpu_architecture
        print "===================================================================="
        print(np.array(matrices[gpu_architecture]))
        missing_entries = []
        for p_drop in p_drops:
            mses = []
            for trial in range(num_trials):
                matrix = np.array(matrices[gpu_architecture])

                matrix_with_missing_entries = matrix.copy()
                (m, n) = matrix_with_missing_entries.shape
                for i in range(m):
                    for j in range(n):
                        if random.uniform(0, 1) < p_drop:
                            matrix_with_missing_entries[(i, j)] = np.NaN
                            matrix_with_missing_entries[(j, i)] = np.NaN
                            missing_entries.append((i, j))
                mses.append(mse(matrix, matrix_with_missing_entries,
                                missing_entries))
            print "Average MSE with p_{drop}=%.1f: %.3f" % (p_drop,
                                                            np.array(mses).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform matrix completion on parsed matrices')
    parser.add_argument('-f', "--filename", required=True,
                        help="Input file which contains performance matrices")
    parser.add_argument('-p', "--p_drops", nargs='+', type=float,
                        help="List of all p_drops to evaluate")
    parser.add_argument('-n', "--num_trials", default=10, type=int,
                        help="Number of trials to perform")
    args = parser.parse_args()

    matrix_completion(args.filename, args.p_drops,
                      args.num_trials)
