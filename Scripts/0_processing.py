import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from pathlib import Path


# Load Yelp dataset (4 feather parts)
data_path = Path(__file__).parent
parts = [pd.read_feather(data_path / f"yelp_part{i}.feather") for i in range(1, 5)]
reviews = pd.concat(parts, ignore_index=True)
reviews["text"] = reviews["text"].astype(str)

# Add review length
if "review_length" not in reviews.columns:
    reviews["review_length"] = reviews["text"].apply(lambda x: len(x.split()))

# Identify empty reviews
reviews["is_empty"] = reviews["text"].apply(lambda x: len(x.strip()) == 0)
print("Empty reviews:", sum(reviews["is_empty"]))
print("Non-empty reviews:", sum(~reviews["is_empty"]))

# Pre-truncate text to avoid >512 token errors
reviews["short_text"] = reviews["text"].apply(lambda x: x[:800])

# Save processed data
out_file = data_path / "yelp_processed.feather"
reviews.to_feather(out_file)
print(f"Saved processed dataset to {out_file}")

# Plot 1: Number of Reviews by Star Rating
sns.countplot(x=reviews["stars"])
plt.title("Number of Reviews by Star Rating (All Reviews)")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
# plt.savefig("plot1_all_reviews.png")

# Plot 2: Average Review Length by Star Rating
avg_lengths = reviews.groupby("stars")["review_length"].mean().reset_index()
sns.barplot(x="stars", y="review_length", data=avg_lengths)
plt.title("Average Review Length by Star Rating")
plt.xlabel("Star Rating")
plt.ylabel("Avg. Review Length (words)")
# plt.savefig("plot2_avg_length.png")

# Plot 3: Distribution of Review Lengths
sns.histplot(reviews["review_length"], bins=50, color="purple")
plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length (words)")
plt.ylabel("Number of Reviews")
# plt.savefig("plot3_length_distribution.png")