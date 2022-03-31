import os
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# use terminal to ls files under this directory
# train_path = "../resource/lib/publicdata/aclImdb/train/"
train_path = "aclimdb/train"

# test data for grade evaluation
# test_path = "../resource/lib/publicdata/imdb_te.csv"
test_path = "imdb_te.csv"

# stopwords to remove from text files
stopwords = []

corpus = []
sentiments = []
test_reviews = []


def imdb_data_preprocess(inpath):
    pos_files = inpath + "/pos"
    neg_files = inpath + "/neg"
    texts = []
    polarities = []

    print("processing positive reviews")
    for filename in os.listdir(pos_files):
        f = open(os.path.join(pos_files, filename), 'r')
        text = remove_stop_words(f.read())
        texts.append(text)
        polarities.append(1)
        f.close()

    print("processing negative reviews")
    for filename in os.listdir(neg_files):
        f = open(os.path.join(neg_files, filename), 'r')
        text = remove_stop_words(f.read())
        texts.append(text)
        polarities.append(0)
        f.close()

    df1 = pd.DataFrame({'text': texts, 'polarity': polarities})
    df1.to_csv("imdb_tr.csv")

    print("processing training data")
    df2 = pd.read_csv("imdb_tr.csv", encoding="ISO-8859-1")
    reviews = df2.values.tolist()

    for review in reviews:
        sentiment = int(review[2])
        sentiments.append(sentiment)
        text = review[1]
        corpus.append(text)

    print("processing testing data")
    df3 = pd.read_csv(test_path, encoding="ISO-8859-1")
    test_cases = df3.values.tolist()

    for case in test_cases:
        text = case[1]
        test_reviews.append(remove_stop_words(text))


def remove_stop_words(in_text):
    words = in_text.split()
    useful_words = []

    for word in words:
        if not word.isnumeric():
            lower_word = word.lower()
            if lower_word not in stopwords:
                useful_words.append(lower_word)

    return " ".join(useful_words)


def unigramSGD(outpath):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    local_vocab = vectorizer.vocabulary_.copy()

    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(vectors, sentiments)

    test_vectorizer = CountVectorizer(vocabulary=local_vocab)
    test_vectors = test_vectorizer.fit_transform(test_reviews)
    predictions = clf.predict(test_vectors)

    np.savetxt(outpath, predictions, fmt="%s")


def bigramSGD(outpath):
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    vectors = vectorizer.fit_transform(corpus)
    local_vocab = vectorizer.vocabulary_.copy()

    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(vectors, sentiments)

    test_vectorizer = CountVectorizer(
        vocabulary=local_vocab, ngram_range=(2, 2))
    test_vectors = test_vectorizer.fit_transform(test_reviews)
    predictions = clf.predict(test_vectors)

    np.savetxt(outpath, predictions, fmt="%s")


def unigram_tfidf_SGD(outpath):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    local_vocab = vectorizer.vocabulary_.copy()

    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(vectors, sentiments)

    test_vectorizer = TfidfVectorizer(vocabulary=local_vocab)
    test_vectors = test_vectorizer.fit_transform(test_reviews)
    predictions = clf.predict(test_vectors)

    np.savetxt(outpath, predictions, fmt="%s")


def bigram_tfidf_SGD(outpath):
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    vectors = vectorizer.fit_transform(corpus)
    local_vocab = vectorizer.vocabulary_.copy()

    clf = SGDClassifier(loss="hinge", penalty="l1")
    clf.fit(vectors, sentiments)

    test_vectorizer = TfidfVectorizer(
        vocabulary=local_vocab, ngram_range=(2, 2))
    test_vectors = test_vectorizer.fit_transform(test_reviews)
    predictions = clf.predict(test_vectors)

    np.savetxt(outpath, predictions, fmt="%s")


if __name__ == "__main__":
    unigramout = "unigram.output.txt"
    unigramtfidfout = "unigramtfidf.output.txt"
    bigramout = "bigram.output.txt"
    bigramtfidfout = "bigramtfidf.output.txt"

    # populate the stopwords array with words to remove from review texts
    f = open("stopwords.en.txt")
    stopwords = f.read().split("\n")
    f.close()

    print("pre-processing data")
    # preprocess the database to create imdb_tr.csv file
    imdb_data_preprocess(train_path)

    print("running unigram SGD")
    # run unigram SGD classifier
    unigramSGD(unigramout)

    print("running bigram SGD")
    # run bigram SGD classifier
    bigramSGD(bigramout)

    print("running unigram SGD with tf-idf")
    # run unigram tf-idf SGD classifier
    unigram_tfidf_SGD(unigramtfidfout)

    print("running bigram SGD with tf-idf")
    # run bigram tf-idf SGD classifier
    bigram_tfidf_SGD(bigramtfidfout)
