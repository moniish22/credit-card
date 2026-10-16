"""Creates sample_transactions.csv (a few fraud + a few legitimate rows) from your dataset,
so the app can offer 'Load an example' buttons.

    python make_samples.py "F:\\path\\to\\creditcard.csv"      (optional 2nd argument: rows per class, default 5)
"""
import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) < 2:
    sys.exit('Usage: python make_samples.py "path\\to\\creditcard.csv" [rows_per_class]')

n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
df = pd.read_csv(sys.argv[1])
if "Class" not in df.columns:
    sys.exit("This file has no 'Class' column (0 = legitimate, 1 = fraud).")

fraud = df[df["Class"] == 1].sample(min(n, (df["Class"] == 1).sum()), random_state=1)
legit = df[df["Class"] == 0].sample(n, random_state=1)
out = pd.concat([fraud, legit]).sample(frac=1, random_state=7).reset_index(drop=True)
target = Path(__file__).resolve().parent / "sample_transactions.csv"
out.to_csv(target, index=False)
print(f"Saved {len(out)} example rows ({len(fraud)} fraud, {len(legit)} legitimate) to {target}")
