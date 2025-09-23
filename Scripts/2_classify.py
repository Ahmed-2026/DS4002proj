from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from pathlib import Path

file_path = Path(__file__).parent.parent / "Data" / "yelp_with_sentiment.csv"
reviews = pd.read_csv(file_path)
reviews = reviews[reviews["sentiment_label"] != "ERROR"] # filter out errored sentiments
reviews = reviews.replace({"POSITIVE": 1, "NEGATIVE": -1}) # convert positive/negative labels to ints

print(reviews.head())

# split data with sentiment into testing/training sets
train_ratio = 0.8
train_size = int(train_ratio * len(reviews))
train = reviews[:train_size]
test = reviews[train_size:]

# extract relevant features
train_features = train[["sentiment_label", "sentiment_score"]]
train_labels = train["stars"]
test_features = test[["sentiment_label", "sentiment_score"]]
test_labels = test["stars"]

# test classifier with a range of nodes
max_accuracy = 0
best_nodes = 0
for x in range(2, 25):
    # train decision tree classifier
    print("⏳ Training decision tree...")
    dt = tree.DecisionTreeClassifier(max_leaf_nodes=x)
    dt = dt.fit(train_features, train_labels)

    # test classifier
    print("⏳ Testing decision tree...")
    predictions = dt.predict(test_features)

    # return metrics
    accuracy = accuracy_score(test_labels, predictions)
    # precision = precision_score(test_labels, predictions, zero_division=0)
    # recall = recall_score(test_labels, predictions)
    # f1 = f1_score(test_labels, predictions)
    print(f"Accuracy for {x} nodes:", accuracy)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_nodes = x

print(f"Best accuracy of {max_accuracy} was achieved with {best_nodes} nodes")
# print(f"Precision: ", precision)
# print(f"Recall: ", recall)
# print(f"F1: ", f1)

dt = tree.DecisionTreeClassifier(max_leaf_nodes=best_nodes)
dt = dt.fit(train_features, train_labels)
print(tree.export_text(dt, feature_names = ["sentiment label", "sentiment strength"]))