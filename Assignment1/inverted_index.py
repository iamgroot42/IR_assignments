import preprocess
import json
import os
import math
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from tqdm import tqdm


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.stemmer = PorterStemmer()

    def loadJSON(self, filename, directory):
        with open(filename, 'r') as f:
            self.index = json.load(f)
        documentNames = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                documentNames.append(os.path.join(root, file))
        self.allDocuments = set(documentNames)

    def dumpJSON(self, destination):
         with open(destination, 'w') as f:
             json.dump(self.index, f)

    def construct_index(self, directory):
        documents = []
        documentNames = []
        for root, dirs, files in os.walk(directory):
            for file in tqdm(files):
                dataRecovered = preprocess.readFile(os.path.join(root, file))
                if dataRecovered:
                    documentNames.append(os.path.join(root, file))
                    documents.append(dataRecovered)
        self.allDocuments = set(documentNames)
        self.construct(documents, documentNames)

    def construct(self, documents, documentNames):
        uniqueWords = list(set([word for words in documents
            for word in words]))
        stemmedWords = list(set([self.stemmer.stem(word)
            for word in uniqueWords]))
        self.index = {word:{'documents':[]} for word in stemmedWords}
        for i, document in tqdm(enumerate(documents)):
            for word in document:
                self.index[self.stemmer.stem(word)]['documents'].append(
                documentNames[i])
        for index in self.index:
            self.index[index]['frequency'] = len(self.index[index]['documents'])
            self.index[index]['documents'] = self.index[index]['documents']

    def XorY(self, x, y):
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        candidates = []
        for x in self.index[x_index]['documents']:
            if x not in candidates:
                candidates.append(x)
        for y in self.index[y_index]['documents']:
            if y not in candidates:
                candidates.append(y)
        return candidates

    def XandnotY(self, x, y):
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        candidates = []
        for x in self.index[x_index]['documents']:
            if x not in self.index[y_index]['documents']:
                candidates.append(x)
        return candidates

    def XornotY(self, x, y):
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        candidates = []
        for x in self.allDocuments:
            if x not in self.index[y_index]['documents']:
                candidates.append(x)
        for y in self.index[x_index]['documents']:
            if y not in candidates:
                candidates.append(y)
        return candidates

    def skipPointerExists(self, index, skipLength, length):
        if index % skipLength == 0:
            if index + skipLength < length:
                return index + skipLength
        return False

    def XandY(self, x, y, sLength=None):
        # Implement skip pointers
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        x_candidates = sorted(self.index[x_index]['documents'])
        y_candidates = sorted(self.index[y_index]['documents'])
        if not sLength:
            skipXlength = int(math.sqrt(self.index[x_index]['frequency']))
            skipYlength = int(math.sqrt(self.index[y_index]['frequency']))
        else:
            skipXlength = sLength
            skipYlength = sLength
        pointerX, pointerY = 0, 0
        candidates = []
        numComparisons = 0
        while pointerX < self.index[x_index]['frequency'] and \
            pointerY < self.index[y_index]['frequency']:
                numComparisons += 1
                skipX = self.skipPointerExists(pointerX, skipXlength, \
                    self.index[x_index]['frequency'])
                skipY = self.skipPointerExists(pointerY, skipYlength, \
                    self.index[y_index]['frequency'])
                if x_candidates[pointerX] == y_candidates[pointerY]:
                    candidates.append(x_candidates[pointerX])
                    pointerX += 1
                    pointerY += 1
                elif x_candidates[pointerX] < y_candidates[pointerY]:
                    if skipX:
                        if x_candidates[skipX] <= y_candidates[pointerY]:
                            while(x_candidates[skipX] <=
                                y_candidates[pointerY] and skipX):
                                pointerX = skipX
                                numComparisons += 1
                                skipX = self.skipPointerExists(pointerX, \
                                  skipXlength,self.index[x_index]['frequency'])
                        else:
                            pointerX += 1
                    else:
                        pointerX += 1
                else:
                    if skipY:
                        if x_candidates[pointerX] >= y_candidates[skipY]:
                            while(x_candidates[pointerX] >=
                                y_candidates[skipY] and skipY):
                                pointerY = skipY
                                numComparisons += 1
                                skipY = self.skipPointerExists(pointerY, \
                                  skipYlength,self.index[y_index]['frequency'])
                        else:
                            pointerY += 1
                    else:
                        pointerY += 1
        return candidates, numComparisons


def generateWordcloud(invertedIndex):
    frequencyDict = {}
    for key in invertedIndex.index:
        frequencyDict[key] = invertedIndex.index[key]['frequency']
    wordcloud = WordCloud().generate_from_frequencies(frequencyDict)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('wordCloud.png')


if __name__ == "__main__":
    import sys
    ii = InvertedIndex()
    if os.path.exists(sys.argv[2]):
        ii.loadJSON(sys.argv[2], sys.argv[1])
    else:
        ii.construct_index(sys.argv[1])
        ii.dumpJSON(sys.argv[2])
    generateWordcloud(ii)
    while True:
        print("Enter query terms (space separated)")
        try:
            x, y = input().split(' ')
        except:
            print("Exiting tool")
            break
        print("Enter operation (AND:1, OR:2, AND NOT:3, OR NOT:4)")
        choice = int(input())
        if choice == 1:
            answer = ii.XandY(x,y)
            print(x," AND ",y,": ", len(answer[0]), ", comparisons : ",answer[1])
        elif choice == 2:
            answer = ii.XorY(x,y)
            print(x," OR ",y," : ", len(answer))
        elif choice == 3:
            answer = ii.XandnotY(x,y)
            print(x," AND NOT ",y," : ", len(answer))
        elif choice == 4:
            answer = ii.XornotY(x,y)
            print(x," OR NOT ",y," : ", len(answer))
