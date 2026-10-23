# Code for Lab 3 Part 2
# Written by: Cody Watkins
# Date: 10-22-2026
# AI usage: I used the built in AI to handle the comments and the grid search portion of the code. I also had it check my code for errors and make suggestions.

#Questions
#1. My classifier predicted positive reviews the best, correctly identifying 392 out of 474.
#  Negative reviews were the hardest to identify, with only 175 out of 299 correctly predicted.
#  Neutral reviews also had a lower F1-score because they were sometimes confused with positive and negative reviews.

#2. The most common mistake was predicting neutral reviews as positive.
#  It also predicted 102 negative reviews as neutral.
#  These mistakes can make sense because neutral reviews can contain both positive and negative words,
#  making it harder for the classifier to tell the difference.

#3. I think the classifier can't reach 100% accuracy because movie reviews can be complicated.
#  A lot of people have different opinions about the same movies, and some reviews can be very subjective or sarcastic.

import os

CRITICS = ["Dennis+Schwartz", "James+Berardinelli", "Scott+Renshaw", "Steve+Rhodes"]


def read_lines(path):
    lines = []
    with open(path, encoding="latin-1") as file:
        for line in file:
            lines.append(line.strip())
    return lines


def load_reviews(folder):
    texts = []
    labels = []
    for critic in CRITICS:
        text_lines = read_lines(os.path.join(folder, critic, "subj." + critic))
        label_lines = read_lines(os.path.join(folder, critic, "label.3class." + critic))
        for i in range(len(text_lines)):
            texts.append(text_lines[i])
            labels.append(int(label_lines[i]))
    return texts, labels


texts, labels = load_reviews("scaledata")
print(len(texts), "reviews")
print("Negative:", labels.count(0), "Neutral:", labels.count(1), "Positive:", labels.count(2))

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# Classification pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(random_state=42))
])

# Train classifier
model.fit(X_train, y_train)

print("\nClassifier trained successfully!")

# Calculate accuracy
from sklearn.metrics import accuracy_score

predicted = model.predict(X_test)

accuracy = accuracy_score(y_test, predicted)

print("\nClassifier Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")

print(classification_report(
    y_test,
    predicted,
    target_names=["Negative", "Neutral", "Positive"]
))

# Confusion matrix
from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted))

# Grid search
from sklearn.model_selection import GridSearchCV

parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__min_df': [1, 2, 3],
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(
    model,
    parameters,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score:")
print(grid_search.best_score_)

# Test the best classifier found by grid search
best_model = grid_search.best_estimator_

best_predicted = best_model.predict(X_test)

best_accuracy = accuracy_score(y_test, best_predicted)

print("\nImproved Classifier Accuracy:")
print(best_accuracy)
print("Improved Accuracy Percentage:", best_accuracy * 100, "%")

print("\nImproved Classification Report:")
print(classification_report(
    y_test,
    best_predicted,
    target_names=["Negative", "Neutral", "Positive"]
))

print("\nImproved Confusion Matrix:")
print(confusion_matrix(y_test, best_predicted))