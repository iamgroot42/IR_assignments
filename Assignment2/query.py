import preprocess
import tf_idf
from autocorrect import spell


class QueryAnswering(object):
    def __init__(self, dataSource, cacheCount=20):
        self.dataSource = dataSource
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
        processedTerm = preprocess.processText(queryTerm)
        processedTerm = self.spellCorrect(processedTerm)
        #Check if query exists in cache
        cachedEntry = self.maybeFetchFromCache(processedTerm)
        Return from cache if exists
        if cachedEntry:
            print("Fetched from Cache")
            return cachedEntry
        else:
            # Compute answer, push into cache
            response = []
            response.append(self.dataSource.query(processedTerm, cosine=False))[:5]
            response.append(self.dataSource.query(processedTerm, cosine=True))[:5]
            self.pushCache(processedTerm, response)
            return response


if __name__ == "__main__":
    tfidf = TfIdf(sys.argv[1], "index.html", 0.75)
    wa = QueryAnswering(tfidf, 20)
    while True:
        query = input()
        answer_a, answer_b = wa.answerQuery(query)
