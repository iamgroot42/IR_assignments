import numpy as np
import preprocess
from collections import Counter
from scipy import spatial
from tqdm import tqdm
import os


class TfIdf(object):
    def __init__(self, dataDir, indexFile, titleRatio=0.75):
        self.dataDir = dataDir
        self.documents = {}
        self.titles = {}
        self.wordFrequencies = {}
        self.titlewordFrequencies = {}
        self.documentVectors = {}
        self.N = 0
        self.vocabulary = []
        self.titleRatio = titleRatio
        self.fileTitles = preprocess.extractFileTitles(indexFile)
        self.preprocessData(dataDir)
        self.precompute_doc_vectors()

    def preprocessData(self, dataDir):
        for docFile in tqdm(os.listdir(dataDir)[:10]):
            if not os.path.isdir(os.path.join(dataDir,docFile)):
                processedFile = preprocess.readFile(os.path.join(dataDir,docFile))
                words = Counter(processedFile)
                self.vocabulary = list(set(self.vocabulary) | set(words.keys()))
                title_words = Counter(preprocess.processText(self.fileTitles[docFile]))
                self.documents[docFile] = words
                self.titles[docFile] = title_words
                for uniqueWord in words.keys():
                    self.wordFrequencies[uniqueWord] = 1 + self.wordFrequencies.get(uniqueWord, 0)
                for uniqueWord in title_words.keys():
                    self.titlewordFrequencies[uniqueWord] = 1 + self.titlewordFrequencies.get(uniqueWord, 0)
                self.N += 1

    def tf_weight(self, tf):
        if tf > 0:
            return 1 + np.log10(tf)
        return 0

    def idf_weight(self, df):
        return np.log10(self.N / (df + 1))

    def precompute_doc_vectors(self):
        for docName, doc in tqdm(self.documents.items()):
            scoreVector = []
            for word in self.vocabulary:
                tfScore = self.tf_weight(doc.get(word, 0))
                idf_score = self.idf_weight(self.wordFrequencies[word])
                contentScore = tfScore * idf_score
                if self.titles.get(docName, None):
                    titletfScore = self.tf_weight(self.titles[docName].get(word, 0))
                else:
                    titletfScore = 0
                titleIdf_score = self.idf_weight(self.titlewordFrequencies.get(word,0))
                titleScore = titletfScore * titleIdf_score
                scoreVector.append(self.titleRatio * contentScore + (1 - self.titleRatio) * titleScore)
            self.documentVectors[docName] = scoreVector


    def query(self, term, cosine):
        retrieved = {}
        indiTerms = term.split(' ')
        docScores = {x:0 for x in self.documents.keys()}
        if cosine:
            query_vector = []
            termFreq = Counter(term)
            for word in self.vocabulary:
                tfScore = self.tf_weight(termFreq.get(word, 0))
                idf_score = self.idf_weight(self.wordFrequencies[word])
                contentScore = tfScore * idf_score
                query_vector.append(contentScore)
            for docName, docVector in self.documentVectors:
                docScores[docName] = 1 - spatial.distance.cosine(docVector, query_vector)
        else:
            for word in indiTerms:
                # IDFCOUNT -> via inverted list
                idf_score = self.idf_weight(self.wordFrequencies[word])
                titleIdf_score = self.idf_weight(self.titlewordFrequencies.get(word, 0))
                for docName, doc in self.documents.items():
                    tfScore = self.tf_weight(doc.get(word, 0))
                    contentScore = tfScore * idf_score
                    if self.titles.get(docName, None):
                        titletfScore = self.tf_weight(self.titles[docName].get(word, 0))
                    else:
                        titletfScore = 0
                    titleScore = titletfScore * titleIdf_score
                    docScores[docName] += self.titleRatio * contentScore + (1 - self.titleRatio) * titleScore
        retrievedDocuments = sorted(dict1, key=docScores.get, reverse=True)
        return retrievedDocuments
