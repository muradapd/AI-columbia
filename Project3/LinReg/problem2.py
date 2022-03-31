import csv
import sys
import numpy as np


def scale(x, m, sd):
    return (x - m) / sd


def build_matrix(in_data):
    matrix = []
    v1 = []
    v2 = []

    for array in in_data:
        v1.append(float(array[0]))
        v2.append(float(array[1]))

    sd1 = np.std(v1)
    m1 = np.mean(v1)
    sd2 = np.std(v2)
    m2 = np.mean(v2)

    for array in in_data:
        f1 = scale(float(array[0]), m1, sd1)
        f2 = scale(float(array[1]), m2, sd2)
        matrix.append([1, f1, f2, float(array[2])])

    return matrix


def loss(vector, matrix, j):
    n = len(matrix)
    i = 0
    sum = 0

    while i < n:
        fx = (vector[0] + (vector[1] * matrix[i][1]) +
              (vector[2] * matrix[i][2]))
        y = matrix[i][3]
        xij = matrix[i][j]
        sum += ((fx - y) * xij)
        i += 1

    return (1/n) * sum


def gradient_descent(matrix, a, i_max):
    i = 0
    b_0 = 0
    b_age = 0
    b_weight = 0
    vector = [b_0, b_age, b_weight]

    while i < i_max:
        for j in range(len(vector)):
            vector[j] -= a * loss(vector, matrix, j)
        i += 1

    return vector


def run(in_data, out_file):
    f = open(out_file, 'w')
    matrix = build_matrix(in_data)
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20]
    iterations = 100
    i = 0

    while i < 10:
        vector = gradient_descent(matrix, alphas[i], iterations)
        f.write(str(alphas[i]) + ',' + str(iterations) +
                ',' + str(vector[0]) + ',' + str(vector[1]) + ',' + str(vector[2]))
        f.write('\n')
        i += 1
    f.close()


def main():
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    with open(str(in_file), newline='') as csvfile:
        in_data = list(csv.reader(csvfile))
    run(in_data, out_file)


if __name__ == '__main__':
    main()
