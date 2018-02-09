import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from num2words import num2words
import string

# Download relevant data
nltk.download('stopwords')
nltk.download('punkt')

# Port Stemmer
STEMMER = PorterStemmer()

def maybeMakeIntoWord(text):
    tbc = text.replace(',','')
    try:
        numword = int(tbc)
        return num2words(numword)
    except:
        try:
            numword = float(tbc)
            return num2words(numword)
        except:
            return text

def stripSpecialCharacters(text):
    return ''.join(ch for ch in text if ch.isalnum() and not ch.isdigit()
        and ch not in string.punctuation)

def processText(text):
    # Convert to lower chatacters
    wordTokens = word_tokenize(text.lower())
    # Remove English stop-words
    stopWords = set(stopwords.words('english'))
    validTokens = list(set(wordTokens) - set(stopWords))
    # Convert numerical figures to words
    validTokens = [maybeMakeIntoWord(x) for x in validTokens]
    # Discard single characters
    validTokens = [x for x in validTokens if len(x) > 1]
    # Remove words with special characters in them, remove special chars
    validTokens = [stripSpecialCharacters(x) for x in validTokens]
    # Stem words
    validTokens = [STEMMER.stem(x) for x in validTokens]
    return validTokens

# Read all files
def readFile(filePath):
    try:
        with open(filePath, 'r', encoding="utf-8", errors='replace') as f:
            corpus = f.read().replace('\n', ' ')
            corpus = processText(corpus)
            return corpus
    except:
        return None


def extractFileTitles(filename):
    titleMapping = {}
    with open(filename, 'r') as f:
        for line in f:
            if "<TR VALIGN=TOP><TD ALIGN=TOP>" in line:
                procText = line.rstrip()
                procText = procText[len("<TR VALIGN=TOP><TD ALIGN=TOP>"):]
                procText = procText[procText.find(">")+1:]
                filename = procText[:procText.find("<")]
                procText = procText[procText.find("</A>  <tab to=T><TD> ") + len("</A>  <tab to=T><TD> "):]
                title = procText[procText.find("<TD> ") + len(("<TD> ")):]
                titleMapping[filename] = title
    return titleMapping
