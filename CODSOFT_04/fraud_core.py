"""Core logic of the fraud-detection app (no Streamlit code here, so it is easy to test).

It adapts to your saved files in the `model/` folder:
    scaler.joblib, feature_columns.joblib, logreg.joblib, dtree.joblib, rf.joblib
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_FILES = {
    "Logistic Regression": "logreg.joblib",
    "Decision Tree": "dtree.joblib",
    "Random Forest": "rf.joblib",
}


class AppError(Exception):
    """A problem with the saved files or the input data, explained in plain words."""


def _as_name_list(obj):
    if isinstance(obj, dict):
        for key in ("columns", "feature_columns", "features"):
            if key in obj:
                obj = obj[key]
                break
    return [str(c) for c in list(obj)]


def load_artifacts(model_dir):
    """Load scaler, feature list and every model that exists in model_dir."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise AppError(f"Folder not found: {model_dir}. Keep the 'model' folder next to app.py.")
    cols_path = model_dir / "feature_columns.joblib"
    scaler_path = model_dir / "scaler.joblib"
    for p in (cols_path, scaler_path):
        if not p.exists():
            raise AppError(f"Missing file: {p.name} (expected inside {model_dir})")
    try:
        columns = _as_name_list(joblib.load(cols_path))
        scaler = joblib.load(scaler_path)
    except Exception as err:
        raise AppError(f"Could not load scaler or feature_columns: {err}") from err

    models, problems = {}, {}
    for label, fname in MODEL_FILES.items():
        path = model_dir / fname
        if not path.exists():
            problems[label] = f"{fname} not found, skipped"
            continue
        try:
            models[label] = joblib.load(path)
        except Exception as err:
            problems[label] = f"could not load {fname}: {err}"
    if not models:
        raise AppError("No model could be loaded. " + "; ".join(problems.values()))
    return {"columns": columns, "scaler": scaler, "models": models, "problems": problems}


def _scaler_columns(scaler, columns):
    """Work out which columns the scaler was fitted on."""
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    n = getattr(scaler, "n_features_in_", None)
    if n is None:
        raise AppError("Cannot tell which columns the scaler was fitted on. Run inspect_model.py.")
    if n == len(columns):
        return list(columns)
    pair = [c for c in columns if c in ("Time", "Amount")]
    if n == len(pair) and n in (1, 2):
        return pair
    raise AppError(
        f"The scaler expects {n} columns but the models use {len(columns)}. "
        "Run `python inspect_model.py` and send me its output."
    )


def prepare_features(df, columns, scaler):
    """Return a numeric, scaled DataFrame with exactly the model's columns."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        raise AppError(f"The data is missing {len(missing)} required column(s): {shown}")
    X = df[columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise AppError("Some values are empty or not numbers. Check the CSV.")
    X = X.astype(float)

    scale_cols = _scaler_columns(scaler, columns)
    absent = [c for c in scale_cols if c not in X.columns]
    if absent:
        raise AppError(f"The scaler needs columns that are not in the features: {absent}")
    if getattr(scaler, "feature_names_in_", None) is not None:
        X[scale_cols] = scaler.transform(X[scale_cols])
    else:
        X[scale_cols] = scaler.transform(X[scale_cols].values)
    return X


def fraud_probability(model, X):
    """Probability (0-1) that each row is fraud."""
    names = getattr(model, "feature_names_in_", None)
    data = X.values
    if names is not None:
        names = [str(n) for n in names]
        if all(n in X.columns for n in names):
            data = X[names]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data)
        classes = list(getattr(model, "classes_", [0, 1]))
        idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, idx]
    if hasattr(model, "decision_function"):
        return 1.0 / (1.0 + np.exp(-model.decision_function(data)))
    return np.asarray(model.predict(data), dtype=float)


def predict_all(df, art):
    """DataFrame with one fraud-probability column per loaded model."""
    X = prepare_features(df, art["columns"], art["scaler"])
    return pd.DataFrame({name: fraud_probability(m, X) for name, m in art["models"].items()})
