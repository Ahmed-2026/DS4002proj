from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from pathlib import Path

# Load dataset with sentiment results
data_path = Path(__file__).parent
reviews = pd.read_feather(data_path / "yelp_with_sentiment.feather")

# Filter out errored sentiments
reviews = reviews[reviews["sentiment_label"] != "ERROR"]

# Convert sentiment labels to numeric
reviews = reviews.replace({"POSITIVE": 1, "NEGATIVE": -1})

print("Data ready for classification:")
print(reviews.head())

# Split into train/test
train_ratio = 0.8
train_size = int(train_ratio * len(reviews))
train = reviews[:train_size]
test = reviews[train_size:]

# Extract features and labels
train_features = train[["sentiment_label", "sentiment_score"]]
train_labels = train["stars"]
test_features = test[["sentiment_label", "sentiment_score"]]
test_labels = test["stars"]

# Train/test decision tree with different max nodes
max_accuracy = 0
best_nodes = 0
for x in range(2, 25):
    dt = tree.DecisionTreeClassifier(max_leaf_nodes=x, random_state=42)
    dt = dt.fit(train_features, train_labels)

    predictions = dt.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Accuracy for {x} nodes: {accuracy:.4f}")
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_nodes = x

print(f"\nBest accuracy of {max_accuracy:.4f} with {best_nodes} nodes")

# Final tree with best node count
dt = tree.DecisionTreeClassifier(max_leaf_nodes=best_nodes, random_state=42)
dt = dt.fit(train_features, train_labels)
predictions = dt.predict(test_features)

# Report metrics
print("\nClassification Report:")
print(classification_report(test_labels, predictions, digits=4))

print("Decision tree rules:")
print(tree.export_text(dt, feature_names=["sentiment_label", "sentiment_score"]))
