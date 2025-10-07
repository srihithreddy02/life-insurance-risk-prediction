# Model Card — Life Insurance Risk Classifier

**Objective:** Predict applicant risk class (0=Low, 1=Medium, 2=High) from synthetic life‑insurance features.

**Algorithms:** Logistic Regression, Random Forest (default).  
**Data:** 5,000 synthetic rows generated with reasonable distributions, no PII.  
**Key Features:** age, bmi, smoker, family_history, medical_score, activity_risk, credit_score, coverage_amount, etc.

**Intended Use:** Educational demo aligned with enterprise insurance analytics (NYL‑style).  
**Limitations:** Synthetic data; not for real underwriting or pricing decisions.

**Metrics (example):** See `reports/confusion_matrix.png` and CLI classification reports.
