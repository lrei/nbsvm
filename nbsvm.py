#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiclass Naive Bayes SVM (NB-SVM)
https://github.com/lrei/nbsvm

Luis Rei <luis.rei@ijs.si> 
@lmrei
http://luisrei.com

Learns a multiclass (OneVsRest) classifier based on word ngrams.
Uses scikit learn. Reads input from TSV files.

Licensed under a Creative Commons Attribution-NonCommercial 4.0 
International License.

Based on a work at https://github.com/mesnilgr/nbsvm:
Naive Bayes SVM by Grégoire Mesnil
"""

import sys
import os
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i + gram])]
    return tokens


def build_counters(filepath, grams, text_row, class_row):
    """Reads text from a TSV file column creating an ngram count 
    Args:
        filepath, the tsv filepath
        grams, the n grams to use
        text_row, row in the tsv file where the text is stored
        class_row, row in the tsv file where the class is stored
    """
    counters = {}

    with open(filepath) as tsvfile:
        for line in tsvfile:
            row = line.split('\t')
            c = int(row[class_row])
            # Select class counter
            if c not in counters:
                # we don't have a counter for this class
                counters[c] = Counter()
            counter = counters[c]

            # update counter
            counter.update(tokenize(row[text_row], grams))

    return counters


def compute_ratios(counters, alpha=1.0):
    """Computes the log-likelihood ratios for each class
    """
    ratios = dict()

    # create a vocabulary - a list of all ngrams
    all_ngrams = set()
    for counter in counters.values():
        all_ngrams.update(counter.keys())
    all_ngrams = list(all_ngrams)
    v = len(all_ngrams)  # the ngram vocabulary size

    # a standard NLP dictionay (ngram -> index map) use to update the 
    # one-hot vector p
    dic = dict((t, i) for i, t in enumerate(all_ngrams))

    # for each class we create calculate a ratio (r_c)
    for c in counters.keys():
        p_c = np.full(v, alpha)
        counter = counters[c]

        for t in all_ngrams:
            p_c[dic[t]] += counter[t]

        # normalize (l1 norm)
        p_c /= np.linalg.norm(p_c, ord=1)
        ratios[c] = np.log(p_c / (1 - p_c))
        
    return dic, ratios, v


def count_lines(data_file):
    """Counts the number of lines in a file
    """
    lines = 0
    with open(data_file) as f:
        for line in f:
            lines += 1
    return lines


def load_data(data_path, text_row, class_row, dic, v, ratios, grams):
    """Create Train or Test matrix and Ground Truth Array
    """
    n_samples = count_lines(data_path)
    n_r = len(ratios)
    classes = ratios.keys()
    Y_real = np.zeros(n_samples, dtype=np.int64)

    # One X (sample) matrix and binary Y (truth) per class
    X = dict()
    Y = dict()
    data = dict()
    indptr = [0]
    indices = []
    for c in classes:
        Y[c] = np.zeros(n_samples, dtype=np.int64)
        data[c] = []

    with open(data_path) as tsvfile:
        n = 0
        for line in tsvfile:
            row = line.split('\t')
            try:
                t = int(row[class_row])
            except:
                print n
                print class_row
                print data_path
                sys.exit(0)

            for c in classes:
                Y[c][n] = int(c == t)
            Y_real[n] = t

            ngrams = tokenize(row[text_row], grams)
            for g in ngrams:
                if g in dic:
                    index = dic[g]
                    indices.append(index)
                    for c in classes:
                        # X[c][n][idx] = ratios[c][idx]
                        data[c].append(ratios[c][index])
            indptr.append(len(indices))

            n += 1
    
    for c in classes:
        X[c] = csr_matrix((data[c], indices, indptr), shape=(n_samples, v),
                          dtype=np.float32)

    return X, Y, Y_real


def main(train, test, text_row, class_row, ngram):
    print('Building vocab, computing ratios')
    ngram = [int(i) for i in ngram]
    counters = build_counters(train, ngram, text_row, class_row)
    dic, ratios, v = compute_ratios(counters)
    classes = ratios.keys()
    print v
    
    print('Loading Data')
    Xs_train, Ys_train, y_train = load_data(train, text_row, 
                                                     class_row, dic, v, 
                                                     ratios, ngram)
    Xs_test, Ys_test, y_true = load_data(test, text_row, class_row, 
                                                  dic, v, ratios, ngram)

    print('Training Classifiers')
    print('classes in train: %d' % len(set(y_train)))
    print('classes in test: %d' % len(set(y_true)))

    svms = dict()
    for c in classes:
        svms[c] = LinearSVC()
        svms[c].fit(Xs_train[c], Ys_train[c])

    print('Testing')
    preds = dict()
    for c in classes:
        preds[c] = svms[c].decision_function(Xs_test[c])

    # not calculate the argmax
    pred = np.zeros(len(y_true))
    for idx in range(0, len(y_true)):
        max_score = float('-inf')
        for c in classes:
            if preds[c][idx] > max_score:
                max_score = preds[c][idx]
                pred[idx] = c

    # finally the scores
    acc_svm = accuracy_score(y_true, pred)
    print('NBSVM: %f' % (acc_svm,))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NB-SVM on some text files.')
    parser.add_argument('--train', required=True, help='path of the train tsv')
    parser.add_argument('--test', required=True, help='path of the test tsv')
    parser.add_argument('--text_row', required=True, 
                        type=int, help='row number of the text')
    parser.add_argument('--class_row', required=True, 
                        type=int, help='row number of the class')
    parser.add_argument('--ngram', required=True,
                        help='N-grams considered e.g. 123 is uni+bi+tri-grams')
    args = vars(parser.parse_args())

    main(**args)
