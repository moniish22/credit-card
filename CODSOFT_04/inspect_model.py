"""Shows what is inside your saved model files and which library versions to use.
Run from this folder:   python inspect_model.py
If anything fails, copy the whole output and send it to your helper."""
import sys
import warnings
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

print("Python:", sys.version.split()[0])
try:
    import sklearn, numpy, pandas
    print("scikit-learn:", sklearn.__version__, "| numpy:", numpy.__version__,
          "| pandas:", pandas.__version__, "| joblib:", joblib.__version__)
except Exception as err:
    print("Could not import libraries:", err)

print("\nFiles in", MODEL_DIR)
if not MODEL_DIR.exists():
    sys.exit("model folder not found. Run this script from the folder that contains 'model'.")
for p in sorted(MODEL_DIR.glob("*")):
    print(f"  {p.name:26s} {p.stat().st_size / 1024:10.1f} KB")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    objs = {}
    for p in sorted(MODEL_DIR.glob("*.joblib")):
        try:
            objs[p.stem] = joblib.load(p)
        except Exception as err:
            print(f"\n[!] could not load {p.name}: {err}")

for w in caught:
    msg = str(w.message)
    if "version" in msg.lower():
        print("\n[!] VERSION WARNING:", msg)
        print("    -> install the version named after 'from version', e.g. pip install scikit-learn==X.Y.Z")
        break

cols = objs.get("feature_columns")
if cols is not None:
    cols = list(cols)
    print(f"\nfeature_columns: {len(cols)} columns\n  {cols}")

sc = objs.get("scaler")
if sc is not None:
    print("\nscaler:", type(sc).__name__, "| n_features_in_ =", getattr(sc, "n_features_in_", "?"))
    names = getattr(sc, "feature_names_in_", None)
    if names is not None:
        print("  fitted on columns:", list(names))

for key, obj in objs.items():
    if key in ("feature_columns", "scaler"):
        continue
    print(f"\n{key}: {type(obj).__name__} | n_features_in_ = {getattr(obj, 'n_features_in_', '?')}"
          f" | classes_ = {list(getattr(obj, 'classes_', []))}"
          f" | has predict_proba = {hasattr(obj, 'predict_proba')}")

print("\nCopy-paste this line into requirements.txt:")
try:
    print(f"scikit-learn=={sklearn.__version__}   (change it if a VERSION WARNING appeared above)")
except Exception:
    pass
