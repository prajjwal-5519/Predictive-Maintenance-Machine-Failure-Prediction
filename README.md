[README.md](https://github.com/user-attachments/files/33394615/README.md)
# Predictive Maintenance: Naive Bayes (Streamlit app)

Predicts machine failure from temperature, speed, torque and tool-wear sensors.
Models: Gaussian Naive Bayes and Kernel (KDE) Naive Bayes, 80/20 split, with Accuracy,
Confusion Matrix heat map, Type I / Type II error, Precision, Recall and F1.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `ml_core.py` | Model, training, threshold tuning, metrics |
| `requirements.txt` | Libraries Streamlit Cloud installs |
| `ai4i2020.csv` | **You add this** (download from UCI / Kaggle) |

## Run locally (optional test)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a **public GitHub repo** and upload `app.py`, `ml_core.py`, `requirements.txt` and `ai4i2020.csv` to the repo root.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file path `app.py`.
4. Click **Deploy**. After 2-3 minutes you get a public link like `https://your-app-name.streamlit.app`.

Any `git push` to the repo redeploys the app automatically.
