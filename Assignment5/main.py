import numpy as np
import os
import argparse
from collections import Counter
from tf_idf import TfIdf

import gensim
import preprocess
from kmeans import kmeans

parser = argparse.ArgumentParser()
parser.add_argument("datapath", type=str, help="path to folder containing training data")
parser.add_argument("keepratio", type=float, help="ratio of features to use, scored by tf-idf")
parser.add_argument("K", type=int, help="Value of K to be used for k-means clustering")


def load_data(dirFolder, featureKeepRatio=1.0):
    classes = sorted(os.listdir(dirFolder))
    vocabulary = set()
    cMap = {i:classes[i] for i in range(len(classes))}
    allDocs = []
    for i, dclass in enumerate(classes):
        documents = os.listdir(os.path.join(dirFolder, dclass))
        np.random.shuffle(documents)
        allDocs.append(documents)
        # Process documents for vocabulary selection
        tfidf = TfIdf(os.path.join(dirFolder, dclass), documents, featureKeepRatio)
        selectedWords = tfidf.selectWords()
        vocabulary = vocabulary | selectedWords
    # Featurize data according to above vocabulary
    vocabulary = list(vocabulary)
    X = []
    for i, dclass in enumerate(classes):
        for doc in allDocs[i]:
            processedFile = preprocess.readFile(os.path.join(os.path.join(dirFolder, dclass), doc))
            words = Counter(processedFile)
            features = [ words.get(w, 0) for w in vocabulary]
            X.append(features)
    return np.stack(X)


def load_data_word2vec(dirFolder, featureKeepRatio=1.0):
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    classes = sorted(os.listdir(dirFolder))
    vocabulary = set()
    cMap = {i:classes[i] for i in range(len(classes))}
    allDocs = []
    for i, dclass in enumerate(classes):
        documents = os.listdir(os.path.join(dirFolder, dclass))
        np.random.shuffle(documents)
        allDocs.append(documents)
        # Process documents for vocabulary selection
        tfidf = TfIdf(os.path.join(dirFolder, dclass), documents, featureKeepRatio)
        selectedWords = tfidf.selectWords()
        vocabulary = vocabulary | selectedWords
    # Featurize data according to above vocabulary
    vocabulary = list(vocabulary)
    X = []
    def getIt(dablu):
        try:
            return model[dablu]
        except:
            return np.zeros((300,))

    for i, dclass in enumerate(classes):
        for doc in allDocs[i]:
            processedFile = preprocess.readFile(os.path.join(os.path.join(dirFolder, dclass), doc))
            words = list(set(processedFile))
            features = [ getIt(w) for w in vocabulary]
            X.append(features)
    return np.stack(X)

def KM(X, K):
	N = X.shape[0] - 1
	np.random.shuffle(X)
	initial_centroids = X[:K]
	loss = kmeans(X, initial_centroids)
	return loss

if __name__ == "__main__":
	args = parser.parse_args()
	# BOW features:
	X_train = load_data(args.datapath, args.keepratio)
	# Word2Vec features
	#X_train = load_data_word2vec(args.datapath, args.keepratio)
	print(X_train.shape[1], "vocab size")
	print(X_train.shape[0], "number of points")
	loss = KM(X_train, args.K)

