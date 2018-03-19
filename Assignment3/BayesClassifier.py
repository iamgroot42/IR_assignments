import numpy as np
from sklearn.metrics import confusion_matrix


class Bayes(object):
    def __init__(self):
        self.condProbs = []

    def train(self, X, Y):
        self.numFeatures = len(X[0])
        temp = sorted(list(set(Y)))
        self.classes = {temp[i]:i for i in range(len(temp))}
        self.posPerFeature = [ [] for _ in range(len(X[0]))]
        self.classProbs = [ 0 for _ in self.classes.keys()]
        for x in X:
            for i, feature in enumerate(x):
                self.posPerFeature[i].append(feature)
        # Find unique possibilities per feature
        for i, z in enumerate(self.posPerFeature):
            temp = list(set(z))
            self.posPerFeature[i] = {temp[i]:i for i in range(len(temp))}
        for _ in self.classes.keys():
            self.condProbs.append([ \
                [ 0 for _ in  self.posPerFeature[i].keys()] \
                    for _ in range(self.numFeatures) ])
        # Calculate P(Class)
        for key in self.classes.keys():
            self.classProbs[self.classes[key]] = float(Y.count(key)) / len(Y)
        for i, x in enumerate(X):
            for j, feature in enumerate(x):
                self.condProbs[self.classes[Y[i]]][j][self.posPerFeature[j][feature]] += 1
        for i in self.classes.keys():
            for j in range(self.numFeatures):
                total = sum(self.condProbs[self.classes[i]][j])
                for k, feature in enumerate(self.condProbs[self.classes[i]][j]):
                    self.condProbs[self.classes[i]][j][k] /= float(total)

    def predict(self, X):
        predictions = []
        for x in X:
            classes = []
            for c in sorted(self.classes.keys()):
                classPrior = self.classProbs[self.classes[c]]
                temp = 1
                for i, feature in enumerate(x):
                    try:
                        temp *= self.condProbs[self.classes[c]][i][self.posPerFeature[i][feature]]
                    except:
                        temp = 0
                classes.append(temp)
            try:
                classes = [ q / float(sum(classes)) for q in classes]
            except:
                pass
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

