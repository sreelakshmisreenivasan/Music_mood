from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import joblib
import re
from sklearn.naive_bayes import GaussianNB

def dummy_fun(doc):
    return doc


def preprocessor(data):
    # Remove unwanted charecters
    contents = re.sub(r'[\W]', ' ',data)
    contents = re.sub("\d+", "", contents)

    # Remove low length words
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    contents = shortword.sub('', contents)

    # Token generation
    tocken = word_tokenize(contents)

    # POS tagging
    txt_pos = [token for token, pos in pos_tag(tocken) if not pos.startswith('NNP')]
    # print(txt_pos)

    # Stop words removel
    stop_words = set(stopwords.words('english'))
    txt = [w for w in txt_pos if not w in stop_words]

    # Stemming
    ps = PorterStemmer()
    stemmed_out = [ps.stem(w) for w in txt]
    # print(filtered_sentence)
    return stemmed_out

vectorizer = joblib.load('vectorizer.pkl')
selector = joblib.load('selector.pkl')

f=open("sample.txt","r")
contents =f.read()
contents = contents.strip()

data = preprocessor(contents)
X = vectorizer.transform([data])

tfs_array = X.todense()

y_pred= selector.predict(tfs_array)
print(contents)
print('\n\nOUTPUT:\n')
print('\t\t',(y_pred[0]).upper())
