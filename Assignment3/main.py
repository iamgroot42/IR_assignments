from  BayesClassifier import Bayes
import numpy as np
import os
import argparse
from collections import Counter
from tf_idf import TfIdf
import preprocess


parser = argparse.ArgumentParser()
parser.add_argument("testratio", type=float, help="percentage of data to be used for testing")
parser.add_argument("datapath", type=str, help="path to folder containint training data")


def load_data(dirFolder, testRatio, featureKeepRatio=1.0):
    classes = sorted(os.listdir(dirFolder))
    vocabulary = set()
    cMap = {i:classes[i] for i in range(len(classes))}
    allDocs = []
    for i, dclass in enumerate(classes):
        documents = os.listdir(os.path.join(dirFolder, dclass))
        np.random.shuffle(documents)
        splitPoint = int(testRatio * len(documents))
        trainDocs, testDocs = documents[splitPoint:], documents[:splitPoint]
        allDocs.append([trainDocs, testDocs])
        # Process documents for vocabulary selection
        tfidf = TfIdf(os.path.join(dirFolder, dclass), trainDocs, featureKeepRatio)
        selectedWords = tfidf.selectWords()
        vocabulary = vocabulary | selectedWords
    # Featurize data according to above vocabulary
    vocabulary = list(vocabulary)
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    for i, dclass in enumerate(classes):
        for j in range(len(allDocs[i])):
            for doc in allDocs[i][j]:
                processedFile = preprocess.readFile(os.path.join(os.path.join(dirFolder, dclass), doc))
                words = Counter(processedFile)
                features = [ words.get(w, 0) for w in vocabulary]
                if j == 0:
                    X_train.append(features)
                    Y_train.append(i)
                else:
                    X_test.append(features)
                    Y_test.append(i)
    return (X_train, Y_train), (X_test, Y_test)


if __name__ == "__main__":
    args = parser.parse_args()
    (X_train, Y_train), (X_test, Y_test) = load_data(args.datapath, args.testratio)
    print(len(X_train), len(X_train[0]), len(Y_train), Y_train[0])
    baCl = Bayes()
    baCl.train(X_train, Y_train)
    confMatrix = baCl.getConfusionMatrix(X_test, Y_test)
    print(confMatrix)
