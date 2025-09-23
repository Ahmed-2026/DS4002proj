import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from pathlib import Path

# load dataset
file_path = Path(__file__).parent.parent / "Data" / "yelp_sample_250k.csv"
reviews = pd.read_csv(file_path)
reviews["text"] = reviews["text"].astype(str)

# Add review length if not already present
if "review_length" not in reviews.columns:
    reviews["review_length"] = reviews["text"].apply(lambda x: len(x.split()))

# Identify empty reviews
reviews["is_empty"] = reviews["text"].apply(lambda x: len(x.strip()) == 0)
print("Empty reviews:", sum(reviews["is_empty"]))
print("Non-empty reviews:", sum(~reviews["is_empty"]))

# Pre-truncate text to avoid >512 token errors
reviews["short_text"] = reviews["text"].apply(lambda x: x[:800])  # ~800 chars ≈ safe for BERT