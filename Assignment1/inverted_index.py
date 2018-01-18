import preprocess
from nltk.stem import PorterStemmer
from tqdm import tqdm
import json
import os
import math


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
                documentNames.append(file)
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
                    documentNames.append(file)
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
        candidates = list(set(self.index[x_index]['documents'])
            | set(self.index[y_index]['documents']))
        return candidates

    def XandYslow(self, x, y):
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        candidates = list(set(self.index[x_index]['documents'])
            & set(self.index[y_index]['documents']))
        return candidates

    def XandnotY(self, x, y):
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        candidates = list(set(self.index[x_index]['documents'])
            - set(self.index[y_index]['documents']))
        return candidates

    def XornotY(self, x, y):
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        candidates = list((self.allDocuments
            - set(self.index[y_index]['documents']))
            | set(self.index[x_index]['documents']))
        return candidates

    def skipPointerExists(self, index, skipLength, length):
        if index % skipLength == 0:
            if index + skipLength < length:
                return index + skipLength
        return False

    def XandY(self, x, y):
        # Implement skip pointers
        x_index = self.stemmer.stem(x.lower())
        y_index = self.stemmer.stem(y.lower())
        x_candidates = sorted(self.index[x_index]['documents'])
        y_candidates = sorted(self.index[y_index]['documents'])
        skipXlength = int(math.sqrt(self.index[x_index]['frequency']))
        skipYlength = int(math.sqrt(self.index[y_index]['frequency']))
        pointerX, pointerY = 0, 0
        candidates = []
        while pointerX < self.index[x_index]['frequency'] and \
            pointerY < self.index[y_index]['frequency']:
                # print(pointerX, pointerY)
                skipX = self.skipPointerExists(pointerX, skipXlength, \
                    self.index[x_index]['frequency'])
                skipY = self.skipPointerExists(pointerX, skipYlength, \
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
                                skipX = self.skipPointerExists(pointerX, \
                                  skipXlength,self.index[x_index]['frequency'])
                        else:
                            pointerX += 1
                    else:
                        pointerX += 1
                else:
                    if skipY:
                        if x_candidates[pointerX] >= y_candidates[skipY]:
                            while(x_candidates[pointerX] <=
                                y_candidates[skipY] and skipY):
                                pointerY = skipY
                                skipY = self.skipPointerExists(pointerY, \
                                  skipYlength,self.index[y_index]['frequency'])
                        else:
                            pointerY += 1
                    else:
                        pointerY += 1
        return candidates


if __name__ == "__main__":
    import sys
    ii = InvertedIndex()
    if os.path.exists(sys.argv[2]):
        ii.loadJSON(sys.argv[2], sys.argv[1])
    else:
        ii.construct_index(sys.argv[1])
        ii.dumpJSON(sys.argv[2])
    x = ii.XandY("read","book")
    print("read AND book : ", len(x))
    x = ii.XorY("read","book")
    print("read OR book : ", len(x))
    x = ii.XandnotY("read","book")
    print("read AND NOT book : ", len(x))
    x = ii.XornotY("read","book")
    print("read OR NOT book : ", len(x))
