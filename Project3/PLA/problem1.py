import csv
import sys


def perceptron(in_data, out_file):
    f = open(out_file, 'w')
    convergence = False
    i = 0
    w1 = 0
    w2 = 0
    b = 0
    w_b = [w1, w2, b]

    while not convergence:
        a1 = int(in_data[i][0])
        a2 = int(in_data[i][1])
        tl = int(in_data[i][2])

        pl = ((w1 * a1) + (w2 * a2) + b)
        if pl > 0:
            pl = 1
        else:
            pl = -1

        if tl * pl <= 0:
            b = b + (tl)
            w1 = w1 + (tl * a1)
            w2 = w2 + (tl * a2)

        i += 1
        if i == 16:
            c_w_b = [w1, w2, b]
            f.write(str(w1) + ',' + str(w2) + ',' + str(b))
            f.write('\n')
            if c_w_b == w_b:
                convergence = True
            else:
                w_b = c_w_b
            i = 0
    f.close()


def main():
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    with open(str(in_file), newline='') as csvfile:
        in_data = list(csv.reader(csvfile))
    perceptron(in_data, out_file)


if __name__ == '__main__':
    main()
