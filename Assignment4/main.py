import numpy as np
import os
import argparse
from collections import Counter
from tf_idf import TfIdf
import preprocess


parser = argparse.ArgumentParser()
parser.add_argument("testratio", type=float, help="percentage of data to be used for testing")
parser.add_argument("datapath", type=str, help="path to folder containing training data")
parser.add_argument("keepratio", type=float, help="ratio of features to use, scored by tf-idf")


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
    return (np.stack(X_train), Y_train), (np.stack(X_test), Y_test)


def KM(X_train, Y_train, X_test, Y_test, k):
	acc = [0 for _ in k]
	for j, x in enumerate(X_test):
		distances = [np.linalg.norm(p-x) for p in X_train]
		for pp, single_k in enumerate(k):
			nearest_ones = np.argsort(distances)[:single_k]
			c = Counter([ Y_train[i] for i in nearest_ones]).most_common(1)[0][0]
			if c == Y_test[j]:
				acc[pp] += 1
	acc = [float(x) / len(X_test) for x in acc]
	return acc


if __name__ == "__main__":
	args = parser.parse_args()
	(X_train, Y_train), (X_test, Y_test) = load_data(args.datapath, args.testratio, args.keepratio)
	X_train = [x/np.linalg.norm(x) for x in X_train]
	X_test = [x/np.linalg.norm(x) for x in X_test]
	X_centroid, Y_centroid = [], []
	for c in np.unique(Y_train):
		indices = np.where(Y_train==c)[0]
		relevant_ones = [ X_train[i] for i in indices]
		X_centroid.append(np.average(relevant_ones, axis=0))
		Y_centroid.append(c)
	rochio_acc = KM(X_centroid, Y_centroid, X_test, Y_test, [1])
	acc = KM(X_train, Y_train, X_test, Y_test, [1, 3, 5])
	print("K-means Accuracy:", acc)
	print("Rocchio accuracy:", rochio_acc)
