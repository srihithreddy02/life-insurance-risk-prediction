from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from utils import CATEGORICAL_COLS, NUMERIC_COLS, TARGET

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", StandardScaler(), NUMERIC_COLS)
    ])

def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def train_val_test_split(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    X, y = split_xy(df)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=rel_val, stratify=y_train_val, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
