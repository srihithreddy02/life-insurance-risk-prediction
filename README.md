# Life Insurance Risk Prediction (NYL-Inspired)

This project predicts the **risk level** of life-insurance applicants (Low / Medium / High) using a small, synthetic dataset. I built it to show how I approach data prep, modeling, and evaluation in an insurance context similar to my work at **New York Life Insurance**. The data is synthetic (no real customer information) but shaped to reflect common signals an analyst would see.

## What I built
- A clean **scikit-learn pipeline** with preprocessing (one-hot encoding + scaling) and two models (Logistic Regression, Random Forest).
- Simple **train/evaluate** scripts so anyone can reproduce results in minutes.
- A **confusion matrix** image and a short **model card** for transparency.
- Clear project structure so the repo is easy to read and reuse.

## Data (synthetic)
The dataset has ~5k rows with features like:
- Demographics: age, gender, region  
- Financials & policy: annual_income, coverage_amount, policy_type, term_years  
- Risk/behavior: bmi, smoker, medical_score, activity_risk, credit_score, claim_history_count, late_payment_count  
- Target: `risk_class` (0=Low, 1=Medium, 2=High)

It’s generated programmatically for learning/demo purposes only.

## How to run
```bash
# (optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Train a model (Random Forest by default)
python src/train_model.py --algorithm rf --out reports/model_rf.joblib

# Evaluate on the test split and save confusion matrix
python src/evaluate.py --model reports/model_rf.joblib --fig reports/confusion_matrix.png
```

## Repo structure
```
data/
  synthetic_life_insurance.csv     # synthetic dataset
src/
  data_prep.py                     # preprocessing + splits
  train_model.py                   # training script (logit / rf)
  evaluate.py                      # evaluation + confusion matrix
  utils.py
reports/
  model_card.md                    # scope, assumptions, limitations
assets/                            # (optional) screenshots/dashboards
README.md
requirements.txt
LICENSE
.gitignore
```

## Results (at a glance)
- The Random Forest baseline gives solid separation across the three risk classes on this synthetic data.
- The confusion matrix (saved in `reports/`) makes it easy to spot where the model confuses Medium vs. High.

## Why this matters
In life insurance analytics, we routinely combine **data quality, feature engineering, reproducible pipelines, and clear reporting**. This repo shows that end-to-end flow in a compact way that’s easy to understand and extend.

## Ideas to extend
- Add a **Power BI** or **Tableau** page showing risk distribution by age, coverage, and smoker status.
- Try **XGBoost** or **LightGBM**, and compare metrics.
- Add **calibration** and **probability thresholds** for operational use cases.
- Log runs and artifacts with MLflow.

## Disclaimer
This is a learning/demo project with synthetic data. It’s **not** intended for real underwriting or pricing decisions.
