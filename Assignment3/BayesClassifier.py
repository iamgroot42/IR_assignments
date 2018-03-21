import numpy as np
from sklearn.metrics import confusion_matrix


class Bayes(object):
    def __init__(self):
        self.condProbs = []

    def train(self, X, Y):
        self.numFeatures = len(X[0])
        temp = sorted(list(set(Y)))
        self.classes = {temp[i]:i for i in range(len(temp))}
        self.classPriors = [ 0 for _ in self.classes.keys()]
        # Initialize with 1s (eqivalent to 1-smoothing)
        self.classConds = [ [1 for _ in range(len(X[0]))] for _ in self.classes.keys()]
        # Calculate class conditionals
        for i, x in enumerate(X):
            for j, feature in enumerate(x):
                self.classConds[self.classes[Y[i]]][j] += feature
        # Normalize counts to get probabilities
        for i in range(len(X[0])):
            denom = 0
            for j in range(len(self.classes)):
                denom += self.classConds[j][i]
            for j in range(len(self.classes)):
                self.classConds[j][i] /= float(denom)
        # Calculate P(Class)
        for key in self.classes.keys():
            self.classPriors[self.classes[key]] = float(Y.count(key)) / len(Y)
        print("Probabilities calculated")

    def predict(self, X):
        predictions = []
        for x in X:
            classes = []
            for c in sorted(self.classes.keys()):
                temp = np.log(self.classPriors[self.classes[c]])
                # Apply 1 smoothing
                for i, feature in enumerate(x):
                    temp += np.log(self.classConds[self.classes[c]][i] * (feature != 0) + 1 * (feature == 0))
                classes.append(temp)
            classes = [ q / float(sum(classes)) for q in classes]
            predictions.append(classes)
        return predictions

    def getModifiedPredictions(self, X, Y):
        argMaxpredictions = []
        predictions = self.predict(X)
        for p in predictions:
            argMaxpredictions.append(np.argmax(p))
        Y_ = [ self.classes[y] for y in Y]
        return (argMaxpredictions, Y_)

    def getSingletonProbs(self, Y):
        Y_ = []
        for p in Y:
            Y_.append(np.max(p))
        return Y_

    def getConfusionMatrix(self, X, Y):
        (argMaxpredictions, Y_) = self.getModifiedPredictions(X, Y)
        cm = confusion_matrix(Y_, argMaxpredictions)
        return cm

