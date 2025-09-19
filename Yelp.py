import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reviews = pd.read_csv("yelp_sample_250k copy.csv")

reviews["text"] = reviews["text"].astype(str)
reviews["is_empty"] = reviews["text"].apply(lambda x: len(x.strip()) == 0)

# Plot 1: All reviews
sns.countplot(x="stars", data=reviews, palette="viridis")
plt.title("Number of Reviews by Star Rating (250k Reviews)")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.savefig("plot1_all_reviews.png", dpi=300)
plt.show()

# Plot 2: Average review length by star rating
avg_length = reviews.groupby("stars")["review_length"].mean().reset_index()
sns.barplot(x="stars", y="review_length", data=avg_length, palette="viridis")
plt.title("Average Review Length by Star Rating")
plt.xlabel("Star Rating")
plt.ylabel("Avg. Review Length (words)")
plt.savefig("plot2_avg_length.png", dpi=300)
plt.show()

# Plot 3: Non-empty text reviews
non_empty_reviews = reviews[reviews["is_empty"] == False]
sns.countplot(x="stars", data=non_empty_reviews, palette="crest")
plt.title("Number of Reviews by Star Rating (Non-Empty Text Only)")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.savefig("plot3_nonempty_reviews.png", dpi=300)
plt.show()

# Plot 4: Distribution of review lengths
sns.histplot(reviews["review_length"], bins=100, kde=False, color="purple")
plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length (words)")
plt.ylabel("Number of Reviews")
plt.xlim(0, 500)  # focus on normal range, avoid extreme outliers
plt.savefig("plot4_length_distribution.png", dpi=300)
plt.show()


print("Empty reviews:", sum(reviews["is_empty"]))
print("Non-empty reviews:", sum(~reviews["is_empty"]))



