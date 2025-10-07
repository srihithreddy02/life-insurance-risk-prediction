# Life Insurance Risk Prediction — NYL‑Style (Ready to Upload)

A complete, **ready-to-run** project that predicts life‑insurance applicant **risk class** (Low/Medium/High) using a synthetic dataset and production‑style Python pipelines. Designed to mirror enterprise analytics work at a life insurer (e.g., New York Life) while remaining fully open-source.

## ✨ What’s inside
- **Synthetic dataset** (`data/synthetic_life_insurance.csv`, 5,000 rows)
- **Clean ML pipeline** with preprocessing (OHE + scaling) and models (Logistic Regression, Random Forest)
- **Train & evaluate** via simple CLI commands
- **Confusion matrix** image artifact saved to `reports/`
- **Model card** documenting assumptions and limitations
- **MIT license** and `.gitignore` included

## 🗂️ Repository structure
```
life-insurance-risk-nyl/
├── data/
│   └── synthetic_life_insurance.csv
├── src/
│   ├── data_prep.py
│   ├── evaluate.py
│   ├── train_model.py
│   └── utils.py
├── reports/
│   └── model_card.md
├── assets/
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## 🚀 Quickstart
```bash
# 1) Create venv and install deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Train a model (Random Forest is default)
python src/train_model.py --algorithm rf --out reports/model_rf.joblib

# 3) Evaluate on the test set & save confusion matrix
python src/evaluate.py --model reports/model_rf.joblib --fig reports/confusion_matrix.png
```

## 📊 Data dictionary (selected)
| Column | Description |
|---|---|
| age | Applicant age |
| gender | Male/Female |
| bmi | Body Mass Index |
| smoker | Yes/No |
| family_history | None/Moderate/High |
| annual_income | USD |
| policy_type | Term/Whole/Universal |
| term_years | If Term policy, 10–30; else 0 |
| coverage_amount | USD coverage |
| medical_score | 20–100 (higher is better health) |
| activity_risk | 0–100 (higher is riskier hobbies) |
| credit_score | 500–850 |
| claim_history_count | # of past claims |
| late_payment_count | Past late payments |
| annual_premium_usd | Premium heuristic |
| risk_class | **Target**: 0=Low, 1=Medium, 2=High |

## 🧪 Notes
- The dataset is synthetic — safe to publish.  
- You can also **swap in your own CSV** with the same columns to re-train.  
- To add a dashboard (Power BI/Tableau), export predictions and create visuals; commit `.pbix` or `.twbx` under `assets/`.

## 📄 Model Card
See [`reports/model_card.md`](reports/model_card.md).

---

Built to showcase life‑insurance analytics skills (feature engineering, modeling, evaluation) aligned with enterprise work.
