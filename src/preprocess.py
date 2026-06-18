# src/preprocess.py

"""
Reusable preprocessing pipeline for the Churn Predictor.
Used by: train.py (training) and main.py (API prediction).

Usage: from src.preprocess import build_preprocessor, load_preprocessor
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Column definitions
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

COLUMNS_TO_DROP = ['customerID']
TARGET_COLUMN = 'Churn'
TARGET_MAP = {'Yes': 1, 'No': 0}


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an unfitted ColumnTransformer pipeline.
    Call .fit_transform(X_train) to fit it.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            drop='if_binart',
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list:
    """
    Return human-readable feature names after OHE expansion.
    Required for SHAP plots and model explainability.
    :param fitted_preprocessor:
    :return: list of feature names
    """
    ohe_names = fitted_preprocessor\
        .named_transformers_['cat']['onehot']\
        .get_feature_names_out(CATEGORICAL_FEATURES).to_list()

    return NUMERIC_FEATURES + ohe_names


def prepare_dataframe(df: pd.DataFrame) -> tuple:
    """
    Clean a raw DataFrame and return X, y ready for splitting.
    Steps: drop ID → encode target → separate features.
    :param df:
    :return:
    """
    df = df.copy()
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP).astype(int)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X,y


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    """Save a fitted preprocessor to disk."""
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved → {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a fitted preprocessor from disk."""
    return joblib.load(path)
