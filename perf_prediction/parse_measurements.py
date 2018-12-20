import argparse
import csv
import numpy as np
import pickle

GPU_ARCHITECTURES = ["K80", "P100", "V100"]

def parse_measurements_file(filename):
    is_next_line_header = False
    gpu_architecture = None
    header = None
    app_times = {}
    app_combinations_times = {}
    with open(filename, 'r') as f:
        for line in csv.reader(f):
            if line[0] in GPU_ARCHITECTURES:
                gpu_architecture = line[0]
                app_times[gpu_architecture] = {}
                app_combinations_times[gpu_architecture] = {}
                is_next_line_header = True
            elif is_next_line_header:
                header = line
                is_next_line_header = False
            else:
                is_all_spaces = all([x == '' for x in line])
                if not is_all_spaces:
                    app_name = line[0]
                    app_time = float(line[1])
                    app_times[gpu_architecture][app_name] = app_time
                    for (other_app_name, line_elem) in zip(header[2:], line[2:]):
                        try:
                            app_combinations_times[gpu_architecture][(app_name,
                                other_app_name)] = [float(x) for x in line_elem.split(", ")]
                        except:
                            continue

    return app_times, app_combinations_times

def compute_matrix(app_combinations_times, app_times):
    flattened_matrices = {}
    for gpu_architecture in app_combinations_times:
        matrix = {}
        applications = app_times[gpu_architecture].keys()
        for app_names in app_combinations_times[gpu_architecture]:
            times = app_combinations_times[gpu_architecture][app_names]
            matrix[app_names] = sum(
                times[i] / app_times[gpu_architecture][app_names[i]] for i in range(len(times)))
        flattened_matrix = []
        for i in range(len(applications)):
            flattened_matrix.append([])
            for j in range(len(applications)):
                flattened_matrix[-1].append(0.0)
        for i in range(len(applications)):
            for j in range(len(applications)):
                if (applications[i], applications[j]) in matrix:
                    flattened_matrix[i][j] = matrix[(applications[i], applications[j])]
                elif (applications[j], applications[i]) in matrix:
                    flattened_matrix[i][j] = matrix[(applications[j], applications[i])]
        flattened_matrices[gpu_architecture] = flattened_matrix
    return flattened_matrices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse a given measurements file')
    parser.add_argument('-f', "--filename", required=True,
                        help="Input file to parse")
    parser.add_argument('-o', "--output_filename", required=True,
                        help="Output file to dump matrices to")
    args = parser.parse_args()

    app_times, app_combinations_times = parse_measurements_file(args.filename)
    flattened_matrices = compute_matrix(app_combinations_times, app_times)
    pickle.dump(flattened_matrices, open(args.output_filename, 'wb'))
