"""Credit card fraud detection app.   Run:  python -m streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import fraud_core as fc

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
SAMPLE_FILE = HERE / "sample_transactions.csv"
RESULTS_FILE = HERE / "results.json"
PLACEHOLDER = "-- choose an example --"

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    return fc.load_artifacts(MODEL_DIR)


@st.cache_data
def get_samples():
    return pd.read_csv(SAMPLE_FILE) if SAMPLE_FILE.exists() else None


st.title("💳 Credit Card Fraud Detector")
st.caption("Educational demo built for the CODSOFT internship. Not for real financial decisions.")

try:
    art = get_artifacts()
except fc.AppError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # e.g. scikit-learn version mismatch
    st.error(f"Could not load the model files: {err}")
    st.info("Run `python inspect_model.py` and install the scikit-learn version it recommends.")
    st.stop()

columns, models = art["columns"], art["models"]

with st.sidebar:
    st.header("Settings")
    names = list(models)
    default = names.index("Logistic Regression") if "Logistic Regression" in names else 0
    model_name = st.selectbox("Model", names, index=default)
    threshold = st.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.05,
                          help="A transaction is flagged as fraud when its probability is at or above this value.")
    st.caption("Lower threshold = catches more fraud but raises more false alarms. "
               "Higher threshold = fewer false alarms but misses more fraud.")
    if art["problems"]:
        with st.expander("Models not loaded"):
            for k, v in art["problems"].items():
                st.write(f"**{k}**: {v}")

tab_single, tab_batch, tab_info = st.tabs(["Single transaction", "Batch (CSV)", "Model info"])

# ---------------------------------------------------------------- single transaction
with tab_single:
    st.subheader("Check one transaction")
    samples = get_samples()
    for c in columns:
        st.session_state.setdefault(f"in_{c}", 0.0)

    if samples is not None and all(c in samples.columns for c in columns):
        def load_example():
            choice = st.session_state.get("example_choice")
            if choice and choice != PLACEHOLDER:
                row = samples.iloc[int(choice.split()[-1]) - 1]
                for c in columns:
                    st.session_state[f"in_{c}"] = float(row[c])
                if "Class" in samples.columns:
                    st.session_state["example_truth"] = "fraud" if int(row["Class"]) == 1 else "legitimate"

        st.selectbox("Load an example transaction (optional)",
                     [PLACEHOLDER] + [f"Example {i + 1}" for i in range(len(samples))],
                     key="example_choice", on_change=load_example)
    else:
        st.info("Tip: run `python make_samples.py <path to creditcard.csv>` to get ready-made examples here.")

    main_cols = [c for c in ("Time", "Amount") if c in columns]
    other_cols = [c for c in columns if c not in main_cols]
    if main_cols:
        for ui, c in zip(st.columns(len(main_cols)), main_cols):
            ui.number_input(c, key=f"in_{c}", format="%.4f")
    with st.expander(f"Other features ({len(other_cols)})"):
        grid = st.columns(4)
        for i, c in enumerate(other_cols):
            grid[i % 4].number_input(c, key=f"in_{c}", format="%.4f")

    if st.button("Check transaction", type="primary"):
        row = pd.DataFrame([{c: st.session_state[f"in_{c}"] for c in columns}])
        try:
            probs = fc.predict_all(row, art).iloc[0]
        except fc.AppError as err:
            st.error(str(err))
        else:
            p = float(probs[model_name])
            st.metric(f"Fraud probability ({model_name})", f"{p:.1%}")
            st.progress(min(max(p, 0.0), 1.0))
            if p >= threshold:
                st.error("🚨 Likely FRAUD")
            else:
                st.success("✅ Looks LEGITIMATE")
            if len(probs) > 1:
                st.write("All models:")
                st.dataframe(probs.map(lambda v: f"{v:.1%}").rename("fraud probability").to_frame())
            if (st.session_state.get("example_choice", PLACEHOLDER) != PLACEHOLDER
                    and "example_truth" in st.session_state):
                st.caption(f"True label of the loaded example: {st.session_state['example_truth']}")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Check many transactions")
    st.write("Upload a CSV with these columns (extra columns such as `Class` are allowed):")
    st.code(", ".join(columns))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
            probs = fc.predict_all(data, art)
        except fc.AppError as err:
            st.error(str(err))
        except Exception as err:
            st.error(f"Could not read the file: {err}")
        else:
            out = data.copy()
            out["fraud_probability"] = probs[model_name].values
            flagged = out["fraud_probability"] >= threshold
            out["prediction"] = np.where(flagged, "FRAUD", "legitimate")
            a, b, c = st.columns(3)
            a.metric("Transactions", f"{len(out):,}")
            b.metric("Flagged as fraud", f"{int(flagged.sum()):,}")
            c.metric("Share flagged", f"{flagged.mean():.2%}")

            if "Class" in out.columns and out["Class"].isin([0, 1]).all():
                from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
                y, yhat = out["Class"].astype(int), flagged.astype(int)
                st.write("**Performance on this file** (it contains the true `Class`)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{precision_score(y, yhat, zero_division=0):.3f}")
                m2.metric("Recall", f"{recall_score(y, yhat, zero_division=0):.3f}")
                m3.metric("F1", f"{f1_score(y, yhat, zero_division=0):.3f}")
                cm = confusion_matrix(y, yhat, labels=[0, 1])
                st.dataframe(pd.DataFrame(cm, index=["actual legit", "actual fraud"],
                                          columns=["predicted legit", "predicted fraud"]))

            st.write("Highest-risk transactions:")
            st.dataframe(out.sort_values("fraud_probability", ascending=False).head(100))
            st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                               file_name="fraud_predictions.csv", mime="text/csv")

# ---------------------------------------------------------------- info
with tab_info:
    st.subheader("About the models")
    st.write(f"Features used: **{len(columns)}**. Models loaded: **{', '.join(models)}**.")
    if RESULTS_FILE.exists():
        st.write("Saved training results (`results.json`):")
        try:
            res = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            if isinstance(res, dict) and res and all(isinstance(v, dict) for v in res.values()):
                st.dataframe(pd.DataFrame(res).T)
            else:
                st.json(res)
        except Exception as err:
            st.warning(f"Could not read results.json: {err}")
    st.write("Features:")
    st.code(", ".join(columns))
