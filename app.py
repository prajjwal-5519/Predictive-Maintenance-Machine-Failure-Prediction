import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import ml_core as core

st.set_page_config(page_title="Predictive Maintenance | Naive Bayes", page_icon="🛠️", layout="wide")

DATA_PATH = "ai4i2020.csv"


# ---------- data + training (cached so the cloud app stays fast) ----------
@st.cache_data(show_spinner=False)
def load_data(path):
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Training Naive Bayes models...")
def get_artifacts(_df, data_key, bandwidth):
    return core.train_all(_df, bandwidth)


def heatmap(m, title):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    labels = np.array([[f"TN\n{m['TN']}", f"FP (Type I)\n{m['FP']}"],
                       [f"FN (Type II)\n{m['FN']}", f"TP\n{m['TP']}"]])
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred Normal", "Pred Failure"],
                yticklabels=["Actual Normal", "Actual Failure"])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------- load data ----------
st.title("🛠️ Machine Failure Prediction (Predictive Maintenance)")
st.caption("Naive Bayes classifiers on AI4I 2020 sensor data: temperature, speed, torque, tool wear")

if os.path.exists(DATA_PATH):
    df, data_key = load_data(DATA_PATH), "repo_csv"
else:
    up = st.file_uploader("`ai4i2020.csv` not found in the repo. Upload it here:", type="csv")
    if up is None:
        st.stop()
    df, data_key = pd.read_csv(up), f"upload_{up.name}_{up.size}"

try:
    core.find_columns(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------- sidebar ----------
st.sidebar.header("Model settings")
model_name = st.sidebar.selectbox("Classifier", ["Gaussian Naive Bayes", "Kernel (KDE) Naive Bayes"])
bandwidth = 0.3
if model_name.startswith("Kernel"):
    bandwidth = st.sidebar.slider("Kernel bandwidth (h)", 0.1, 1.0, 0.3, 0.05,
                                  help="Small h = wiggly density, large h = smoother density")
art = get_artifacts(df, data_key, bandwidth)

thr_mode = st.sidebar.radio("Decision threshold", ["Default (0.50)", "Tuned (recall-focused)", "Manual"])
if thr_mode.startswith("Default"):
    threshold = 0.5
elif thr_mode.startswith("Tuned"):
    threshold = art["thresholds"][model_name]
    st.sidebar.info(f"Threshold chosen by 5-fold CV on training data (F2-score): **{threshold:.2f}**")
else:
    threshold = st.sidebar.slider("P(Failure) cutoff", 0.01, 0.99, 0.30, 0.01)
st.sidebar.caption(f"Train/Test split: 80/20 → {art['n_train']} / {art['n_test']} rows")

tab_pred, tab_eval, tab_data = st.tabs(["🔮 Predict", "📊 Evaluation", "🗂️ Data"])

# ---------- Predict tab ----------
with tab_pred:
    st.subheader("Enter live sensor readings")
    cols = st.columns(5)
    vals = []
    for c, f in zip(cols, art["feats"]):
        lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
        vals.append(c.number_input(f, min_value=lo, max_value=hi, value=med, step=(hi - lo) / 200))

    p_fail, label = core.predict_one(art, model_name, vals, threshold)
    c1, c2 = st.columns([1, 2])
    c1.metric("P(Failure)", f"{p_fail * 100:.1f}%")
    if label:
        c2.error(f"⚠️ **FAILURE predicted** (P = {p_fail:.3f} ≥ threshold {threshold:.2f}). Schedule maintenance.")
    else:
        c2.success(f"✅ **Normal operation** (P = {p_fail:.3f} < threshold {threshold:.2f}).")
    st.progress(min(max(p_fail, 0.0), 1.0))

# ---------- Evaluation tab ----------
with tab_eval:
    st.subheader(f"{model_name} | threshold = {threshold:.2f} | test set (20%)")
    proba = art["models"][model_name].predict_proba(art["Xte_s"])[:, 1]
    m = core.evaluate(art["y_te"], (proba >= threshold).astype(int))

    k = st.columns(6)
    k[0].metric("Accuracy", f"{m['Accuracy']:.3f}")
    k[1].metric("Precision", f"{m['Precision']:.3f}")
    k[2].metric("Recall", f"{m['Recall']:.3f}")
    k[3].metric("F1-score", f"{m['F1']:.3f}")
    k[4].metric("Type I error", f"{m['TypeI']:.3f}", help="False alarm rate = FP/(FP+TN)")
    k[5].metric("Type II error", f"{m['TypeII']:.3f}", help="Missed failure rate = FN/(FN+TP)")

    left, right = st.columns(2)
    left.pyplot(heatmap(m, "Confusion Matrix Heat Map"))
    right.text(m["report"])
    right.markdown(
        f"- **Type I (false alarms):** {m['FP']} healthy machines flagged as failing\n"
        f"- **Type II (missed failures):** {m['FN']} real failures not detected\n\n"
        "Lowering the threshold ⬇️ reduces Type II but ⬆️ increases Type I.")

    st.subheader("Threshold sweep")
    ths = np.round(np.linspace(0.05, 0.95, 19), 2)
    rows = []
    for t in ths:
        mm = core.evaluate(art["y_te"], (proba >= t).astype(int))
        rows.append({"Threshold": t, "Type I error": mm["TypeI"], "Type II error": mm["TypeII"],
                     "Precision": mm["Precision"], "Recall": mm["Recall"], "F1": mm["F1"]})
    sweep = pd.DataFrame(rows).set_index("Threshold")
    st.line_chart(sweep[["Type I error", "Type II error"]])
    st.dataframe(sweep.style.format("{:.3f}"), use_container_width=True)

# ---------- Data tab ----------
with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(50), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.write("**Class balance**")
    c1.bar_chart(df[art["target"]].value_counts().rename({0: "Normal", 1: "Failure"}))
    c1.caption(f"Failure rate = {art['fail_rate'] * 100:.2f}% (imbalanced → accuracy alone is misleading)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(df[art["feats"] + [art["target"]]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heat map")
    fig.tight_layout()
    c2.pyplot(fig)
