import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import numpy as np

# load dataset
reviews = pd.read_csv("yelp_sample_250k copy.csv")
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

# load sentiment model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.backends.mps.is_available() else -1
)

# Batch sentiment analysis
def analyze_in_batches(texts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment analysis"):
        batch = texts[i:i + batch_size]
        try:
            batch_results = sentiment_analyzer(batch, truncation=True, max_length=256)
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            batch_results = [{"label": "ERROR", "score": np.nan}] * len(batch)
        results.extend(batch_results)
    return results

print("⏳ Running sentiment analysis...")
results = analyze_in_batches(reviews["short_text"].tolist(), batch_size=32)

# Attach results safely
reviews["sentiment_label"] = [r["label"] for r in results]
reviews["sentiment_score"] = [r["score"] for r in results]

# save output
out_file = "yelp_with_sentiment.csv"
reviews.to_csv(out_file, index=False)
print(f"Sentiment analysis complete. Saved to {out_file}")
