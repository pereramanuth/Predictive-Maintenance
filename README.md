# 🛠️ Predictive Maintenance — Early-Warning Failure Classifier

An end-to-end machine learning pipeline that predicts machine failure risk from live sensor
readings, deployed as a FastAPI service with a live dashboard front end.

🔗 **Live demo:** https://melodic-sable-87f9cf.netlify.app/

---

## 🎯 What it does

This is an **early-warning classifier**, not a remaining-useful-life forecaster. Given six
inputs — product type, air temperature, process temperature, rotational speed, torque, and
tool wear — it returns a calibrated failure probability and a 🟢 Low / 🟡 Medium / 🔴 High
risk tag, so maintenance teams can prioritize inspections before a breakdown happens.

Built on the [AI4I 2020 Predictive Maintenance Dataset (UCI)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

---

## 📊 Results

| Metric | Value |
|---|---|
| Model | Gradient Boosting Classifier |
| Test ROC-AUC | 0.964 |
| Test PR-AUC | 0.822 |
| Failure-class precision | 0.75 |
| Failure-class recall | 0.74 |
| Overall accuracy | 0.98 |

⚠️ Threshold (0.885) was tuned on **cross-validated PR curves** — not the test set — to avoid
leakage into the reported metrics. The dataset is heavily imbalanced (~3.4% failure rate),
so PR-AUC and the Failure-class recall/precision matter far more than raw accuracy here.

---

## 🔄 Pipeline

1. 🔍 **`01_eda.ipynb`** : exploratory analysis, class imbalance, feature distributions
2. 🧹 **`02_prepare.ipynb`** : cleaning, label encoding, `StandardScaler`, train/test split
3. 🤖 **`03_model.ipynb`** : baseline model comparison (Logistic Regression, Random Forest, Gradient Boosting)
4. 🎛️ **`04_model_selection.ipynb`** : randomized hyperparameter search, CV-based threshold tuning, final model refinement and export to `model_bundle.pkl`
5. 🚀 **Deployment** : FastAPI service (`app.py`, `inference.py`) + static dashboard frontend

---

## 🏗️ Architecture

```
Sensor reading → FastAPI /predict → inference.py (scaler + label encoder, same as training)
              → Gradient Boosting model → probability → threshold → risk tier → JSON response
```

`inference.py` replicates the Phase 2 preprocessing exactly (same encoder, same scaler, same
column order) so predictions match the training notebooks bit-for-bit , a common silent
failure mode in ML deployment is preprocessing drift between training and serving.

---

## ▶️ Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# Docs: http://127.0.0.1:8000/docs
```

---

## 🧰 Tech stack

Python · scikit-learn · FastAPI · Pydantic · joblib · HTML/CSS/JS dashboard · Hugging Face Spaces (API) · Netlify (frontend)

---

## 📁 Project structure

```
├── 01_eda.ipynb
├── 02_prepare.ipynb
├── 03_model.ipynb
├── 04_model_selection.ipynb
├── app.py
├── inference.py
├── model_bundle.pkl
├── sanity_check.py
├── requirements.txt
└── index.html
```
