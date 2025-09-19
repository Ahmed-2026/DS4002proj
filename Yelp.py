import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

reviews = pd.read_csv("yelp_sample_250k.csv")

reviews["text"] = reviews["text"].astype(str)
reviews["is_empty"] = reviews["text"].apply(lambda x: len(x.strip()) == 0)

# Plot 1: All reviews
sns.countplot(x="stars", data=reviews, palette="viridis")
plt.title("Number of Reviews by Star Rating (250k Reviews)")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.savefig("plot1_all_reviews.png", dpi=300)
plt.show()

# Plot 2: Empty text reviews
empty_reviews = reviews[reviews["is_empty"] == True]
sns.countplot(x="stars", data=empty_reviews, palette="mako")
plt.title("Number of Reviews by Star Rating (Empty Text Only)")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.savefig("plot2_empty_reviews.png", dpi=300)
plt.show()

# Plot 3: Non-empty text reviews
non_empty_reviews = reviews[reviews["is_empty"] == False]
sns.countplot(x="stars", data=non_empty_reviews, palette="crest")
plt.title("Number of Reviews by Star Rating (Non-Empty Text Only)")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.savefig("plot3_nonempty_reviews.png", dpi=300)
plt.show()

print("Empty reviews:", sum(reviews["is_empty"]))
print("Non-empty reviews:", sum(~reviews["is_empty"]))
