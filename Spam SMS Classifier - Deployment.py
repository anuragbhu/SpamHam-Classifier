# Importing essential libraries
import pandas as pd
import pickle

# Loading the dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
df = pd.read_csv('Spam SMS Collection', sep='\t', names=['label', 'message'])

# Importing essential libraries for performing Natural Language Processing on 'SMS Spam Collection' dataset
# Natural Language Toolkit
import nltk
# Regular Expression
import re
# Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query.
nltk.download('stopwords')
# The nltk.corpus package defines a collection of corpus (refers to a collection of texts) reader classes
from nltk.corpus import stopwords
# Stemming is the process of producing morphological variants of a root/base word.
from nltk.stem.porter import PorterStemmer

# Cleaning the messages
# list
corpus = []
# tuples
ps = PorterStemmer()

for i in range(0,df.shape[0]):

  # Cleaning special character from the message
  message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df.message[i])

  # Converting the entire message into lower case
  message = message.lower()

  # Tokenizing the review by words
  words = message.split()

  # Removing the stop words
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in words]

  # Joining the stemmed words
  message = ' '.join(words)

  # Building a corpus of messages
  corpus.append(message)
  
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

# Extracting dependent variable from the dataset
y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))

# Model Building

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-sms-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
