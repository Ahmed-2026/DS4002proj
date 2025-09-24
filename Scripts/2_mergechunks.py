import pandas as pd
from pathlib import Path
import os

data_path = Path(__file__).parent

# Collect all chunk files
chunk_files = sorted(data_path.glob("sentiment_chunk_*.feather"))
print(f"Found {len(chunk_files)} chunk files")

# Merge them all
dfs = [pd.read_feather(f) for f in chunk_files]
full = pd.concat(dfs, ignore_index=True)

# Save final file
out_file = data_path / "yelp_with_sentiment.feather"
full.to_feather(out_file)

# Report stats
file_size_gb = os.path.getsize(out_file) / (1024**3)
print(f"Merged into {out_file}")
print(f"Total rows: {full.shape[0]}")
print(f"File size: {file_size_gb:.2f} GB")

# Optional check: verify row count matches original
print("\nIf row count matches original dataset size, merge was successful")
