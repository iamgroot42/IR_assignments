import numpy as np
import preprocess
from collections import Counter
from scipy import spatial
from tqdm import tqdm
import os


class TfIdf(object):
    def __init__(self, baseDir, docs, ratio):
        self.wordFrequencies = {}
        self.documents = {}
        self.N = 0
        self.wordFrequencies = {}
        self.ratio = ratio
        self.vocabulary = []
        self.docs = docs
        self.baseDir = baseDir
        self.preprocessData()

    def preprocessData(self):
       for doc in self.docs:
            processedFile = preprocess.readFile(os.path.join(self.baseDir, doc))
            words = Counter(processedFile)
            self.vocabulary = list(set(self.vocabulary) | set(words.keys()))
            self.documents[doc] = words
            for uniqueWord in words.keys():
                self.wordFrequencies[uniqueWord] = 1 + self.wordFrequencies.get(uniqueWord, 0)
            self.N += 1

    def tf_weight(self, tf):
        if tf > 0:
            return 1 + np.log10(tf)
        return 0

    def idf_weight(self, df):
        return np.log10(self.N / (df + 1))

    def selectWords(self):
        newVocab = []
        for doc in self.docs:
            scores = []
            for word in self.vocabulary:
                tfScore = self.tf_weight(self.documents.get(word, 0))    
                idf_score = self.idf_weight(self.wordFrequencies[word])
                scores.append(- tfScore * idf_score)
            selection = np.argsort(scores)[:int(len(scores) * self.ratio)]
            [newVocab.append(self.vocabulary[x]) for x in selection]
        return set(newVocab)

