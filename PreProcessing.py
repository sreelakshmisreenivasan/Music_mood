from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import joblib
import re
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

f=open("OutPut2.txt","r",encoding="utf-8")
print('Reading lines')
lines=oneline=f.readlines()
preprocessed=[]
for line in lines:    
    songandlines=()
    #List for the lines    
    songlines=[]
    #Reading song name
    songname=line.split("\t")[1]
    #reading the mode of the song
    moode=line.split("\t")[2]
    #reading the lyrics
    lyrics=line.split("\t")[3].split(".")
    #preprocessing each line of lyrics
    for i in lyrics:
        songlines.append(preprocessor(i))
    songandlines=(songname,moode,songlines)  
    preprocessed.append(songandlines)
    
num_songs = len(preprocessed)

lbl = []
lyric_all = []

print('Obtaining labels...')
for song_name,label,lyrics in preprocessed:
    #print(label,song_name)
    lyric = []
    for l in lyrics:
        lyric.extend(l)
    lbl.append(label)
    lyric_all.append(lyric)    

labels_unique = list(set(lbl))
num_labels = len(labels_unique)
count = [[0]*num_labels][0]
to_show = [[] for x in range(num_labels)]

for i in range(num_songs):
    for c in range(num_labels):
        if lbl[i] == labels_unique[c]:
            count[c] = count[c] + 1
            lyr = " ".join(lyric)
            to_show[c].append(lyr)
        else:
            continue
print('\nTotal number of songs:\t',num_songs)
print('\nNumber of songs in each category:')
for i in range(num_labels):
    print('\n\t\t*',labels_unique[i],':',count[i])

# Display word clouds
neg = to_show[0]
to_show_neg = " ".join(neg)
print('\nShowing WordCloud for:\n\t\t\tNegative')
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt

wc = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10)
wordcloud = wc.generate(to_show_neg) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

# Convert raw frequency counts into TF-IDF (Term Frequency -- Inverse Document Frequency) values
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert raw frequency counts into TF-IDF values
print('\nPerforming tf-idf ...')
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun)
X = tfidf.fit_transform(lyric_all)
print(type(X))
tfs_array = X.todense()
print(tfs_array.shape)

Y = lbl
print('\n\nSplitting trainset and test set')
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
split_ratio = 0.25
# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y,
                                                test_size = split_ratio,
                                                random_state = 42)

Xtrain = Xtrain.toarray()
Xtest = Xtest.toarray()
########################################################################
print('\nPerforming classification.')
print('\nNaive Bayes:')
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()


# Train the model using the training sets
model.fit(Xtrain,Ytrain)

#Predict Output

y_pred= model.predict(Xtest)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
acc1 = metrics.accuracy_score(Ytest, y_pred)
print("\tAccuracy:",acc1)

#########################################################################
print('\n\nRandom Forest')
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(Xtrain,Ytrain)
#print('Done.')

#print('Making predictions using test-set...',end="\t")
y_pred=clf.predict(Xtest)
#print('Done.')

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
acc2 = metrics.accuracy_score(Ytest, y_pred)
print("\tAccuracy:",acc2)
#########################################################################

print('\n\nSupport Vector Machine:')
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(Xtrain, Ytrain)

y_pred = svclassifier.predict(Xtest)
acc3 = metrics.accuracy_score(Ytest, y_pred)
print("\tAccuracy:",acc3)

#########################################################################

import matplotlib.pyplot as plt
import numpy as np

y = [acc1,acc2,acc3]
N = len(y)
x = range(N)
width = 1/2

label = ['Naive Bayes','Random Forest','Support Vector Machine']
index = np.arange(len(label))

plt.bar(x, y, width, color="blue")
plt.xlabel('Classifiers', fontsize=5)
plt.ylabel('Accuracy', fontsize=5)
plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Classifiers Comparison')
plt.show()


joblib.dump(tfidf, 'vectorizer.pkl')
joblib.dump(model, 'selector.pkl')

#Later, I can load it and ready to go:

vectorizer = joblib.load('vectorizer.pkl')
selector = joblib.load('selector.pkl')
