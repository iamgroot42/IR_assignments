from  BayesClassifier import Bayes
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("testratio", type=float, help="percentage of data to be sued for testing")
parser.add_argument("datapath", type=str, help="path to folder containint training data")


def load_data(dirFolder, testRatio):
    classes = sorted(os.listdir(dirFolder))
    cMap = {i:classes[i] for i in range(len(classes))}
    for i, dclass in enumerate(classes):
        documents = os.listdir(os.path.join(dirFolder, dclass))
        np.random.shuffle(documents)
        splitPoint = int(testRatio * len(documents))
        trainDocs, testDocs = documents[splitPoint:], documents[:splitPoint]
    exit()

if __name__ == "__main__":
    args = parser.parse_args()
    (X_train, Y_train), (X_test, Y_test) = load_data(args.datapath, args.testratio)
    baCl = Bayes()
    baCl.train(X, Y)

