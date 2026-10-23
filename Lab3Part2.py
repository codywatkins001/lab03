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

# Split the data
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