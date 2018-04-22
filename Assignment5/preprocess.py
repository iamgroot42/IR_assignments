import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


# Download relevant data
nltk.download('stopwords')
nltk.download('punkt')


def stripSpecialCharacters(text):
    return ''.join(ch for ch in text if ch.isalnum() and not ch.isdigit()
        and ch not in string.punctuation)

def processText(text):
    stopWords = set(stopwords.words('english'))
    wordTokens = word_tokenize(text.lower())
    validTokens = list(set(wordTokens) - set(stopWords))
    # Discard single characters
    validTokens = [stripSpecialCharacters(x) for x in validTokens]
    validTokens = [x for x in validTokens if len(x) > 1]
    # Remove words with special characters in them, remove special chars
    return validTokens


def readFile(filePath):
    try:
        with open(filePath, 'r', encoding="utf-8", errors='replace') as f:
            corpus = f.read().replace('\n', ' ')
            corpus = corpus[corpus.find("Lines:"):] #Data specific, ignore headers
            corpus = processText(corpus)
            return corpus
    except:
        return None


if __name__ == "__main__":
    import sys
    print(readFile(sys.argv[1]))
