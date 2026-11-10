import pandas as pd
import os

# --- EDIT THESE TWO LINES ---
seed = 47                          # e.g. 42 for Group Alpha
reviews_path = "archive\Reviews.csv"     # e.g. "/Users/yourname/Downloads/Reviews.csv"
# ----------------------------

df = pd.read_csv(reviews_path)
sample = df.sample(n=50000, random_state=seed)

out_path = os.path.join(os.path.expanduser("~"), "Downloads", f"group_sample_seed{seed}.csv")
sample.to_csv(out_path, index=False)

print(f"Saved {len(sample)} rows to {out_path}")
print(sample[['Score', 'Text']].head(3))