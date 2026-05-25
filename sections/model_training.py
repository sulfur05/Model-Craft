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
