import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False


def _ensure_preprocessed_data():
    if "preprocessor" not in st.session_state:
        st.info(
            "Run the 'Data Preprocessing' step first to configure preprocessing and split the data."
        )
        return None

    required_keys = ["X_train", "X_test", "y_train", "y_test", "feature_columns"]
    missing = [k for k in required_keys if k not in st.session_state]
    if missing:
        st.error(
            "Some preprocessing outputs are missing: " + ", ".join(missing) + ". "
            "Please re-run the preprocessing step."
        )
        return None

    return {
        "preprocessor": st.session_state["preprocessor"],
        "X_train": st.session_state["X_train"],
        "X_test": st.session_state["X_test"],
        "y_train": st.session_state["y_train"],
        "y_test": st.session_state["y_test"],
        "feature_columns": st.session_state["feature_columns"],
    }
    
def _get_model_options(task_type: str):
    if task_type == "regression":
        models = [
            "Linear Regression",
            "Ridge Regression",
            "Random Forest Regressor",
        ]
        if HAS_XGBOOST:
            models.append("XGBoost Regressor")
    else:
        models = [
            "Logistic Regression",
            "Random Forest Classifier",
        ]
        if HAS_XGBOOST:
            models.append("XGBoost Classifier")
    return models




def _build_model(task_type: str, model_name: str, params: dict):
    if task_type == "regression":
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Ridge Regression":
            return Ridge(alpha=params.get("alpha", 1.0))
        if model_name == "Random Forest Regressor":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42,
            )
        if model_name == "XGBoost Regressor" and HAS_XGBOOST:
            return XGBRegressor(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
            )
    else:
        if model_name == "Logistic Regression":
            return LogisticRegression(
                max_iter=params.get("max_iter", 1000),
                C=params.get("C", 1.0),
                n_jobs=-1,
            )
        if model_name == "Random Forest Classifier":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", None),
                random_state=42,
            )
        if model_name == "XGBoost Classifier" and HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 6),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
    raise ValueError(f"Unsupported model: {model_name}")

