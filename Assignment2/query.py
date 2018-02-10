import preprocess
import tf_idf
from autocorrect import spell


class QueryAnswering(object):
    def __init__(self, dataSource, cacheCount=20):
        self.dataSource = dataSource
        self.cacheCount = cacheCount
        self.cache = []

    def spellCorrect(self, text):
        words = text.split(' ')
        corrected = [spell(x) for x in words]
        return ' '.join(corrected)

    def maybeFetchFromCache(self, queryTerm):
        for term in self.cache:
            if term[0] == queryTerm:
                return term[1]
        return None

    def popCache(self):
        self.cache = self.cache[:-1]

    def pushCache(self, term, response):
        self.cache.append([term, response])
        if len(self.cache) > self.cacheCount:
            self.popCache()

    def answerQuery(self, queryTerm):
        processedTerm = queryTerm.split(' ')
        processedTerm = [self.spellCorrect(x) for x in processedTerm]
        processedTerm = ' '.join(processedTerm)
        processedTerm = preprocess.processText(processedTerm)
        processedTerm = ' '.join(processedTerm)
        # Check if query exists in cache
        cachedEntry = self.maybeFetchFromCache(processedTerm)
        # Return from cache if exists
        if cachedEntry:
            print('\033[93m' + "Cache Hit!" + '\033[0m' )
            return cachedEntry
        else:
            # Compute answer, push into cache
            response = []
            response.append(self.dataSource.query(processedTerm, cosine=False)[:5])
            response.append(self.dataSource.query(processedTerm, cosine=True)[:5])
            self.pushCache(processedTerm, response)
            return response


if __name__ == "__main__":
    import sys
    tfidf = tf_idf.TfIdf(sys.argv[1], "index.html", 0.75)
    wa = QueryAnswering(tfidf, 20)
    while True:
        try:
            query = input("Enter query: ")
            answer_a, answer_b = wa.answerQuery(query)
            print('\033[94m' + "Using pure tf-idf"+ '\033[0m', answer_a)
            print('\033[92m' + "Using cosine similarity" + '\033[0m', answer_b)
        except:
            print('\033[91m' + "Exiting!" + '\033[0m')
            exit()
