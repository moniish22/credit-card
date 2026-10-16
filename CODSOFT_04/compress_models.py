"""Makes the saved models smaller (same predictions) so they fit GitHub's 25 MB browser-upload limit.
Writes compressed copies to the folder 'model_compressed'; your originals are NOT changed.

    python compress_models.py
Then copy the files you want from model_compressed into 'model' (replace the originals).
"""
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
src, dst = HERE / "model", HERE / "model_compressed"
dst.mkdir(exist_ok=True)
for p in sorted(src.glob("*.joblib")):
    obj = joblib.load(p)
    target = dst / p.name
    joblib.dump(obj, target, compress=("xz", 3))
    print(f"{p.name:26s} {p.stat().st_size / 1e6:8.2f} MB  ->  {target.stat().st_size / 1e6:8.2f} MB")
