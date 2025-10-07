import argparse, joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from data_prep import train_val_test_split
from utils import load_data

def main(data_path: str, model_path: str, fig_path: str):
    df = load_data(data_path)
    _, _, X_test, _, _, y_test = train_val_test_split(df)
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    print("Test report:\n", classification_report(y_test, y_pred))
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    print(f"Saved confusion matrix to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_life_insurance.csv")
    parser.add_argument("--model", default="reports/model_rf.joblib")
    parser.add_argument("--fig", default="reports/confusion_matrix.png")
    args = parser.parse_args()
    main(args.data, args.model, args.fig)
