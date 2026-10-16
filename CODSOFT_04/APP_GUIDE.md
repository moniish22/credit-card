# Fraud Detection App: Step-by-Step Guide

Your `CODSOFT_04` project is **credit card fraud detection**. It already has the trained models
(`model/` folder with `logreg`, `dtree`, `rf`, `scaler`, `feature_columns`). This package adds a
web app on top of them. You do **not** retrain anything.

What the app does:
- **Single transaction tab:** type or load one transaction and see the fraud probability.
- **Batch tab:** upload a CSV, get a prediction for every row, and download the results.
- **Model info tab:** shows your `results.json`.
- **Sidebar:** choose the model (Logistic Regression, Decision Tree, Random Forest) and the fraud threshold.

---

## What is in this package

| File | Purpose |
|---|---|
| `app.py` | The web app |
| `fraud_core.py` | Loads your models and makes predictions |
| `inspect_model.py` | Checks your model files and prints the library versions to use |
| `make_samples.py` | Creates example transactions from your dataset |
| `compress_models.py` | Makes big model files smaller for GitHub |
| `requirements.txt` | Libraries the app needs |
| `tests/` | Automatic tests (optional) |
| `.gitignore` | Tells Git which files not to upload |

---

## Step 1: Copy the files into your project folder

Your project folder is `F:\codesoft\CODSOFT_04` (the one that contains `model`, `plots`, `fraud_detection.py`).

1. In that folder, **rename your old `requirements.txt` to `requirements-training.txt`**. (It was for training; the app needs a smaller list.)
2. Unzip this package and **copy everything inside it** into `F:\codesoft\CODSOFT_04`.

Result:
```
CODSOFT_04/
├── model/                    (yours)
├── plots/                    (yours)
├── fraud_detection.py        (yours)
├── results.json              (yours)
├── README.md                 (yours)
├── requirements-training.txt (your old one, renamed)
├── app.py                    (new)
├── fraud_core.py             (new)
├── inspect_model.py          (new)
├── make_samples.py           (new)
├── compress_models.py        (new)
├── requirements.txt          (new)
└── APP_GUIDE.md              (new)
```
`app.py` must sit **next to** the `model` folder.

---

## Step 2: Install what the app needs

Open Command Prompt:
```cmd
cd /d F:\codesoft\CODSOFT_04
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```
You should see `(.venv)` at the start of the line. (PowerShell: use `.venv\Scripts\Activate.ps1`.)

---

## Step 3: Check your model files (important)

```cmd
python inspect_model.py
```

It prints your library versions, the feature columns, and what each saved file contains.

**Look for two things:**

1. **`VERSION WARNING`** in the output. It means your models were saved with a different scikit-learn version, for example *"from version 1.5.0 when using version 1.8.0"*. Fix it by installing the old version:
   ```cmd
   pip install scikit-learn==1.5.0
   ```
   (use the number from the warning). Then run `python inspect_model.py` again until the warning is gone.
2. The last line, **"Copy-paste this line into requirements.txt"**. Open `requirements.txt` in Notepad and replace the scikit-learn line with the one shown. Do this even when there is no warning. It makes the online app match your computer.

If you see any `[!] could not load ...` line, or anything else confusing, copy the whole output and send it to me.

---

## Step 4 (recommended): Make example transactions

You need your original dataset file (usually called `creditcard.csv`, the Kaggle credit card fraud file).

```cmd
python make_samples.py "F:\path\to\creditcard.csv"
```
This creates `sample_transactions.csv` with 5 fraud and 5 normal transactions. The app then shows a **"Load an example"** menu. The file is tiny, so it is safe to upload to GitHub. Do **not** upload the full `creditcard.csv` (about 150 MB).

---

## Step 5: Run the app on your computer

```cmd
python -m streamlit run app.py
```
Your browser opens at `http://localhost:8501`. (If Streamlit asks for an email, just press Enter.) Press `Ctrl + C` in the terminal to stop it.

---

## Step 6: Try it

1. In the **Single transaction** tab, choose **Example 1** from the menu. The 30 numbers fill in automatically.
2. Click **Check transaction**. You get a fraud probability and either "Likely FRAUD" or "Looks LEGITIMATE". Below the result, the app prints the true label of the example, so you can see if the model was right.
3. Repeat with the other examples.
4. Change **Model** in the sidebar (for example to Random Forest) and compare.
5. Move the **Fraud threshold** slider. A lower value catches more fraud but creates more false alarms.
6. In the **Batch (CSV)** tab, upload `sample_transactions.csv`. Because it contains the true `Class`, the app also shows precision, recall, F1 and a confusion matrix.

Write down what you see. It is useful for your report.

---

## Step 7: Upload to GitHub

Your GitHub folder is `CODSOFT_04/CODSOFT_04` (the one in your link).

### 7a. The Random Forest file is too big
`rf.joblib` is about 47 MB. GitHub's browser upload accepts files up to **25 MB**. Two options:

**Option 1 (recommended): shrink it.**
```cmd
python compress_models.py
```
This writes smaller copies (same predictions) into a new folder `model_compressed`. Check the sizes it prints. If `rf.joblib` is now under 25 MB, copy it from `model_compressed` into `model` and replace the old one. The app loads it the same way.

**Option 2: skip it.** Do not upload `rf.joblib`. The app detects this and works with Logistic Regression and Decision Tree only.

### 7b. Upload in the browser
1. Open `github.com/moniish22/CODSOFT_04`, then the `CODSOFT_04` folder.
2. Click **Add file**, then **Upload files**.
3. Drag in: `app.py`, `fraud_core.py`, `requirements.txt`, `sample_transactions.csv`, `APP_GUIDE.md`, `inspect_model.py`, `make_samples.py`, `compress_models.py`.
4. Click **Commit changes**.
5. Open the `model` folder on GitHub and check that these exist: `scaler.joblib`, `feature_columns.joblib`, `logreg.joblib`, `dtree.joblib` (and `rf.joblib` if it is under 25 MB). Add any missing file with **Add file, Upload files**.

Do **not** upload the `.venv` folder or `creditcard.csv`.

---

## Step 8: Put the app online (free)

1. Go to **share.streamlit.io** and click **Continue with GitHub**.
2. Click **Create app**.
3. Fill in:
   - **Repository:** `moniish22/CODSOFT_04`
   - **Branch:** `main`
   - **Main file path:** `CODSOFT_04/app.py`
4. Click **Deploy** and wait a few minutes.

If `app.py` is in the root of your repository instead, use `app.py` as the path.

You get a link like `https://something.streamlit.app`. Add it to your README and your GitHub repository's **About, Website** field.

---

## Step 9: Mention the app in your report

Add a short section: *"A Streamlit web app lets users check single transactions or upload a CSV, choose between Logistic Regression, Decision Tree and Random Forest, and adjust the fraud threshold."* Put in the app link and a screenshot.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named streamlit` | Activate the virtual environment, then `pip install -r requirements.txt` |
| App says `Folder not found ... model` | `app.py` must be in the same folder as `model` |
| App says `Missing file: scaler.joblib` | Check the file names in `model` match exactly |
| `InconsistentVersionWarning` or a load error | Install the scikit-learn version named in the message (Step 3) and put it in `requirements.txt` |
| `The data is missing N required column(s)` | Your CSV needs the same columns as the model (shown on the Batch tab) |
| `The scaler expects N columns but the models use M` | Run `python inspect_model.py` and send me the output |
| Streamlit Cloud: `ModuleNotFoundError` | Add that library to `requirements.txt` and commit |
| Streamlit Cloud: model fails to load | Pin the exact scikit-learn version in `requirements.txt` |
| GitHub says file too large | Compress or skip `rf.joblib` (Step 7a) |
| Online app is asleep | Free apps sleep when unused. Click the wake-up button and wait |

---

## How it works (to explain to your mentor)

1. The app reads `feature_columns.joblib` to know which 30 inputs the models expect.
2. It scales the inputs with `scaler.joblib`, exactly as in training.
3. It passes them to the chosen model and reads the fraud probability with `predict_proba`.
4. If the probability is at or above the threshold (default 50%), the transaction is flagged as fraud.

Fraud is rare in this dataset (about 0.17% of transactions), so precision, recall and F1 matter more than accuracy. A threshold slider lets the user trade missed fraud against false alarms.
