import argparse, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_prep import build_preprocessor, train_val_test_split
from utils import load_data

def build_model(algorithm: str):
    pre = build_preprocessor()
    if algorithm == "logit":
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    elif algorithm == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
    else:
        raise ValueError("algorithm must be 'logit' or 'rf'")
    return Pipeline([("pre", pre), ("clf", clf)])

def main(data_path: str, algorithm: str, out_path: str):
    df = load_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    pipe = build_model(algorithm)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    print("Validation report:\n", classification_report(y_val, y_pred))
    joblib.dump(pipe, out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_life_insurance.csv")
    parser.add_argument("--algorithm", choices=["logit","rf"], default="rf")
    parser.add_argument("--out", default="reports/model_rf.joblib")
    args = parser.parse_args()
    main(args.data, args.algorithm, args.out)
