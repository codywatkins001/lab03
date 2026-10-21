from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Select categories
categories = [
    'alt.atheism',
    'soc.religion.christian',
    'comp.graphics',
    'sci.med'
]

# Loading the 20 newsgroups dataset
twenty_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42
)

# Feature Vector Matrix
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# Calculating Term Frequency and Inverse Document Frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Training a classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

# Load test data
twenty_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=42
)

# Convert test data using training vocabulary
X_test_counts = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Making predictions
predicted = clf.predict(X_test_tfidf)

# Calculate accuracy
accuracy = np.mean(predicted == twenty_test.target)

print("Average accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")