# CODSOFT_04 – Credit Card Fraud Detection

Machine learning internship task: build a model to classify credit card
transactions as **fraudulent** or **legitimate**.

## Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(simulated card transactions, already split into `fraudTrain.csv` and
`fraudTest.csv`).

| | rows | fraud rate |
|---|---|---|
| Train | 1,296,675 | 0.58% |
| Test | 555,719 | 0.39% |

The two files are **not** included in this repo (too large for GitHub) —
download them from Kaggle and place them in the project root, or pass
their paths on the command line.

## Approach

1. **Feature engineering** (`fraud_detection.py::load_and_engineer`)
   - Drop direct identifiers: card number, name, address, transaction ID
     (these would leak identity, not fraud signal, and don't generalize).
   - Extract `hour`, `day_of_week`, `month` from the transaction timestamp.
   - Compute customer `age` from date of birth.
   - Compute `distance_km` between the customer's location and the
     merchant's location (haversine formula) — fraud tends to happen
     further from the cardholder's usual location.
   - Frequency-encode `merchant` (rare merchants can be riskier).
   - One-hot encode `category` (grocery, gas, shopping, etc.), map `gender`
     to 0/1.
2. **Models trained**, all with class-imbalance handling (`class_weight`)
   since fraud is <1% of transactions:
   - Logistic Regression (on standardized features) — simple, interpretable
     baseline.
   - Decision Tree (max_depth=12) — captures non-linear thresholds (e.g.
     "amount > $X and hour < Y").
   - Random Forest (200 trees, max_depth=16) — ensemble of trees, generally
     the strongest of the three for this kind of tabular data.
3. **Evaluation** on the held-out `fraudTest.csv` using metrics suited to
   imbalanced classification (accuracy alone is misleading here):
   precision/recall/F1 for the fraud class, ROC-AUC, and PR-AUC
   (average precision).

## Results

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.04 | 0.74 | 0.07 | 0.909 | 0.121 |
| Decision Tree | 0.13 | 0.95 | 0.23 | 0.976 | 0.676 |
| **Random Forest** | **0.55** | **0.87** | **0.68** | **0.996** | **0.852** |

**Random Forest is the best model**: it catches 87% of fraudulent
transactions while keeping false positives low enough that more than half
of its fraud flags are correct (55% precision) — a much better
precision/recall trade-off than the other two. Logistic Regression, being
a linear model, struggles with the non-linear interactions between amount,
time and location that separate fraud from legitimate spend, so it needs a
very low decision threshold (hurting precision) to catch fraud at all.

Top predictive features (Random Forest): transaction **amount**, **hour**
of day, **merchant frequency**, cardholder **age**, and transaction
**category** (online shopping / grocery / gas).

See `plots/roc_curves.png`, `plots/pr_curves.png`, and
`plots/feature_importance.png` for visuals, and `results.json` for the
full classification reports and confusion matrices.

## Usage

```bash
pip install -r requirements.txt
python fraud_detection.py --train fraudTrain.csv --test fraudTest.csv
```

This retrains all three models and regenerates `plots/`, `model/`
(saved `.joblib` models + scaler), and `results.json`.

To reuse a trained model for new predictions:

```python
import joblib
rf = joblib.load("model/rf.joblib")
cols = joblib.load("model/feature_columns.joblib")
# build a DataFrame with the same engineered columns, then:
rf.predict(X_new[cols])
```

## Repo structure

```
fraud_detection.py     # feature engineering + training + evaluation
requirements.txt
results.json           # metrics for all 3 models
model/                 # saved models (logreg, dtree, rf) + scaler
plots/                 # ROC curve, PR curve, feature importance
```

## Notes / possible extensions

- Random Forest's precision (55%) could be pushed higher by tuning the
  classification threshold or trying gradient-boosted trees
  (XGBoost/LightGBM), at some cost to recall.
- SMOTE or other resampling could be tried as an alternative to
  `class_weight` for handling the imbalance.
- Distance and merchant-frequency features gave a solid lift over using
  only raw transaction fields — worth exploring further engineered
  features (e.g. rolling transaction count/spend per card).
