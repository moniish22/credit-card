"""
Credit Card Fraud Detection
============================
Task: Build a model to detect fraudulent credit card transactions using
Logistic Regression, Decision Tree, and Random Forest classifiers.

Dataset: Kaggle "Credit Card Transactions Fraud Detection Dataset"
         (fraudTrain.csv / fraudTest.csv) - simulated card-present and
         card-not-present transactions labelled as fraud (1) or legit (0).

Usage:
    python fraud_detection.py --train fraudTrain.csv --test fraudTest.csv

Outputs (written to ./plots and ./model):
    - roc_curves.png, pr_curves.png, feature_importance.png
    - logreg.joblib, dtree.joblib, rf.joblib, scaler.joblib, feature_columns.joblib
    - results.json (metrics for all three models)
"""

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Feature engineering
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/long points, in kilometres."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_and_engineer(path, merchant_freq=None):
    """
    Read a raw fraudTrain/fraudTest CSV and turn it into a model-ready
    numeric table. Drops direct identifiers (name, card number, address,
    transaction id) and instead engineers behavioural / contextual signals:
    time of day, customer age, distance between customer and merchant,
    and merchant frequency.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[df.columns[0]])  # unnamed row-index column

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["month"] = df["trans_date_trans_time"].dt.month
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # Frequency-encode merchant (fit on train, reuse on test to avoid leakage)
    if merchant_freq is None:
        merchant_freq = df["merchant"].value_counts(normalize=True)
    df["merchant_freq"] = df["merchant"].map(merchant_freq).fillna(0)

    df["gender"] = df["gender"].map({"M": 0, "F": 1})

    keep_cols = [
        "category", "amt", "gender", "city_pop", "hour", "day_of_week",
        "month", "age", "distance_km", "merchant_freq", "is_fraud",
    ]
    df = df[keep_cols]
    df = pd.get_dummies(df, columns=["category"], drop_first=True)
    return df, merchant_freq


# ---------------------------------------------------------------------------
# 2. Evaluation helper
# ---------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test, results, roc_data, pr_data):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion matrix:\n", cm)
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")

    results[name] = {"report": report, "confusion_matrix": cm.tolist(), "roc_auc": roc_auc, "pr_auc": pr_auc}
    roc_data[name] = roc_curve(y_test, y_proba)[:2]
    pr_data[name] = precision_recall_curve(y_test, y_proba)[:2]


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main(train_path, test_path, out_dir="."):
    plots_dir = f"{out_dir}/plots"
    model_dir = f"{out_dir}/model"
    import os
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Loading & engineering features...")
    train, merchant_freq = load_and_engineer(train_path)
    test, _ = load_and_engineer(test_path, merchant_freq=merchant_freq)

    # Align columns (one-hot categories may differ slightly between splits)
    train, test = train.align(test, join="left", axis=1, fill_value=0)

    X_train = train.drop(columns=["is_fraud"]).astype(float)
    y_train = train["is_fraud"]
    X_test = test.drop(columns=["is_fraud"]).astype(float)
    y_test = test["is_fraud"]

    print("Train shape:", X_train.shape, " Fraud rate:", round(y_train.mean(), 4))
    print("Test shape:", X_test.shape, " Fraud rate:", round(y_test.mean(), 4))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, roc_data, pr_data = {}, {}, {}

    # ---- Logistic Regression (needs scaled features + class_weight for imbalance) ----
    t0 = time.time()
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    print(f"Logistic Regression trained in {time.time()-t0:.1f}s")
    evaluate("Logistic Regression", logreg, X_test_scaled, y_test, results, roc_data, pr_data)

    # ---- Decision Tree ----
    t0 = time.time()
    dtree = DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42)
    dtree.fit(X_train, y_train)
    print(f"Decision Tree trained in {time.time()-t0:.1f}s")
    evaluate("Decision Tree", dtree, X_test, y_test, results, roc_data, pr_data)

    # ---- Random Forest ----
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=16, class_weight="balanced_subsample", n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    print(f"Random Forest trained in {time.time()-t0:.1f}s")
    evaluate("Random Forest", rf, X_test, y_test, results, roc_data, pr_data)

    # ---- Save models ----
    joblib.dump(logreg, f"{model_dir}/logreg.joblib")
    joblib.dump(dtree, f"{model_dir}/dtree.joblib")
    joblib.dump(rf, f"{model_dir}/rf.joblib")
    joblib.dump(scaler, f"{model_dir}/scaler.joblib")
    joblib.dump(list(X_train.columns), f"{model_dir}/feature_columns.joblib")

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Plots ----
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/roc_curves.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 6))
    for name, (prec, rec) in pr_data.items():
        plt.plot(rec, prec, label=f"{name} (AP={results[name]['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/pr_curves.png", dpi=120)
    plt.close()

    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    importances.head(15).sort_values().plot(kind="barh")
    plt.title("Random Forest - Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/feature_importance.png", dpi=120)
    plt.close()

    print("\nTop 10 features (Random Forest):\n", importances.head(10))
    print("\nAll done. Models in ./model, plots in ./plots, metrics in results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--train", default="fraudTrain.csv")
    parser.add_argument("--test", default="fraudTest.csv")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    main(args.train, args.test, args.out)
