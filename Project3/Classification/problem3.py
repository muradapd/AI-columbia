import csv
import sys
import sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def svm_linear(train, test):
    correct1 = 0
    correct2 = 0
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for arr in train:
        train_samples.append([arr[0], arr[1]])
        train_labels.append(arr[2])

    for arr in test:
        test_samples.append([arr[0], arr[1]])
        test_labels.append(arr[2])

    slk = SVC(C=0.1, kernel='linear')
    slk.fit(train_samples, train_labels)

    train_classify = slk.predict(train_samples)
    test_classify = slk.predict(test_samples)

    for i in range(len(train_classify)):
        if train_classify[i] == train_labels[i]:
            correct1 += 1
    best_score = correct1 / len(train)

    for i in range(len(test_classify)):
        if test_classify[i] == test_labels[i]:
            correct2 += 1
    test_score = correct2/len(test)

    return [best_score, test_score]


def svm_polynomial(train, test):
    correct1 = 0
    correct2 = 0
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for arr in train:
        train_samples.append([arr[0], arr[1]])
        train_labels.append(arr[2])

    for arr in test:
        test_samples.append([arr[0], arr[1]])
        test_labels.append(arr[2])

    slk = SVC(C=0.1, kernel='poly', gamma=0.5, degree=5)
    slk.fit(train_samples, train_labels)

    train_classify = slk.predict(train_samples)
    test_classify = slk.predict(test_samples)

    for i in range(len(train_classify)):
        if train_classify[i] == train_labels[i]:
            correct1 += 1
    best_score = correct1 / len(train)

    for i in range(len(test_classify)):
        if test_classify[i] == test_labels[i]:
            correct2 += 1
    test_score = correct2/len(test)

    return [best_score, test_score]


def svm_rbf(train, test):
    correct1 = 0
    correct2 = 0
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for arr in train:
        train_samples.append([arr[0], arr[1]])
        train_labels.append(arr[2])

    for arr in test:
        test_samples.append([arr[0], arr[1]])
        test_labels.append(arr[2])

    slk = SVC(C=10, gamma=6, kernel='rbf')
    slk.fit(train_samples, train_labels)

    train_classify = slk.predict(train_samples)
    test_classify = slk.predict(test_samples)

    for i in range(len(train_classify)):
        if train_classify[i] == train_labels[i]:
            correct1 += 1
    best_score = correct1 / len(train)

    for i in range(len(test_classify)):
        if test_classify[i] == test_labels[i]:
            correct2 += 1
    test_score = correct2/len(test)

    return [best_score, test_score]


def logistic(train, test):
    correct1 = 0
    correct2 = 0
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for arr in train:
        train_samples.append([arr[0], arr[1]])
        train_labels.append(arr[2])

    for arr in test:
        test_samples.append([arr[0], arr[1]])
        test_labels.append(arr[2])

    slk = LogisticRegression(C=100.0)
    slk.fit(train_samples, train_labels)

    train_classify = slk.predict(train_samples)
    test_classify = slk.predict(test_samples)

    for i in range(len(train_classify)):
        if train_classify[i] == train_labels[i]:
            correct1 += 1
    best_score = correct1 / len(train)

    for i in range(len(test_classify)):
        if test_classify[i] == test_labels[i]:
            correct2 += 1
    test_score = correct2/len(test)

    return [best_score, test_score]


def knn_help(train, example):
    k = 5
    neighbors = []
    nearest_neighbors = []
    compare = []

    for i in range(len(train)):
        point1 = np.array((example[0], example[1]))
        point2 = np.array((train[i][0], train[i][1]))
        dist = np.linalg.norm(point1 - point2)
        neighbors.append([i, dist])
    sorted_neighbors = sorted(neighbors, key=lambda x: x[1])
    sorted_neighbors.pop(0)
    for i in range(k):
        nearest_neighbors.append(sorted_neighbors.pop(0))

    positive = 0
    negative = 0

    for neighbor in nearest_neighbors:
        if train[neighbor[0]][2] > 0:
            positive += 1
        else:
            negative += 1

    prediction = 0

    if negative < positive:
        prediction = 1
    compare.append([prediction, int(example[2])])

    return compare


def knn(train, test):
    correct1 = 0
    correct2 = 0
    compare1 = []
    compare2 = []

    for example in train:
        for item in knn_help(train, example):
            compare1.append(item)

    for comp in compare1:
        if comp[0] == comp[1]:
            correct1 += 1
    best_score = correct1/len(train)

    for example in test:
        for item in knn_help(train, example):
            compare2.append(item)

    for comp in compare2:
        if comp[0] == comp[1]:
            correct2 += 1
    test_score = correct2/len(test)

    return [best_score, test_score]


def decision_tree(train, test):
    correct1 = 0
    correct2 = 0
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for arr in train:
        train_samples.append([arr[0], arr[1]])
        train_labels.append(arr[2])

    for arr in test:
        test_samples.append([arr[0], arr[1]])
        test_labels.append(arr[2])

    slk = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    slk.fit(train_samples, train_labels)

    train_classify = slk.predict(train_samples)
    test_classify = slk.predict(test_samples)

    for i in range(len(train_classify)):
        if train_classify[i] == train_labels[i]:
            correct1 += 1
    best_score = correct1 / len(train)

    for i in range(len(test_classify)):
        if test_classify[i] == test_labels[i]:
            correct2 += 1
    test_score = correct2/len(test)

    return [best_score, test_score]


def random_forest(train, test):
    correct1 = 0
    correct2 = 0
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []

    for arr in train:
        train_samples.append([arr[0], arr[1]])
        train_labels.append(arr[2])

    for arr in test:
        test_samples.append([arr[0], arr[1]])
        test_labels.append(arr[2])

    slk = RandomForestClassifier(max_depth=10, min_samples_split=2)
    slk.fit(train_samples, train_labels)

    train_classify = slk.predict(train_samples)
    test_classify = slk.predict(test_samples)

    for i in range(len(train_classify)):
        if train_classify[i] == train_labels[i]:
            correct1 += 1
    best_score = correct1 / len(train)

    for i in range(len(test_classify)):
        if test_classify[i] == test_labels[i]:
            correct2 += 1
    test_score = correct2/len(test)

    print(best_score, test_score)

    return [best_score, test_score]


def run(in_data, out_file):
    f = open(out_file, 'w')
    labels = []
    for array in in_data:
        labels.append(array[2])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        in_data, labels, test_size=0.4, stratify=labels)
    y_train
    y_test

    # scores = svm_linear(x_train, x_test)
    # f.write("svm_linear" + ", " +
    #         str(scores[0]) + ", " + str(scores[1]) + "\n")

    # scores = svm_polynomial(x_train, x_test)
    # f.write("svm_polynomial" + ", " +
    #         str(scores[0]) + ", " + str(scores[1]) + "\n")

    # scores = svm_rbf(x_train, x_test)
    # f.write("svm_rbf" + ", " + str(scores[0]) + ", " + str(scores[1]) + "\n")

    # scores = logistic(x_train, x_test)
    # f.write("logistic" + ", " + str(scores[0]) + ", " + str(scores[1]) + "\n")

    # scores = knn(x_train, x_test)
    # f.write("knn" + ", " + str(scores[0]) + ", " + str(scores[1]) + "\n")

    # scores = decision_tree(x_train, x_test)
    # f.write("decision_tree" + ", " +
    #         str(scores[0]) + ", " + str(scores[1]) + "\n")

    scores = random_forest(x_train, x_test)
    f.write("random_forest" + ", " +
            str(scores[0]) + ", " + str(scores[1]) + "\n")

    f.close()


def main():
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    with open(str(in_file), newline='') as csvfile:
        in_data = list(csv.reader(csvfile))
        in_data.pop(0)
        for array in in_data:
            array[0] = float(array[0])
            array[1] = float(array[1])
            array[2] = int(array[2])
    run(in_data, out_file)


if __name__ == '__main__':
    main()
