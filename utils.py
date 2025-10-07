import pandas as pd

CATEGORICAL_COLS = ["gender","smoker","family_history","policy_type","region"]
NUMERIC_COLS = ["age","bmi","annual_income","term_years","coverage_amount","medical_score","activity_risk","credit_score","claim_history_count","late_payment_count","annual_premium_usd"]
TARGET = "risk_class"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
