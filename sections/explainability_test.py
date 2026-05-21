import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap as sp

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

def _is_tree_model(model):
    tree_models = (RandomForestClassifier, RandomForestRegressor)
    if HAS_XGBOOST:
        tree_models = tree_models + (XGBClassifier, XGBRegressor)
    return isinstance(model, tree_models)


def _is_linear_model(model):
    return isinstance(model, (LinearRegression, Ridge, LogisticRegression))


def _is_classifier_model(model):
    classifier_models = (LogisticRegression, RandomForestClassifier)
    if HAS_XGBOOST:
        classifier_models = classifier_models + (XGBClassifier,)
    return isinstance(model, classifier_models)


def _prepare_shap_input(preprocessor, X_train):
    X_train_proc = preprocessor.transform(X_train)

    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    if X_train_proc.ndim == 1:
        X_train_proc = X_train_proc.reshape(-1, 1)

    if len(feature_names) != X_train_proc.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    X_train_proc = pd.DataFrame(X_train_proc, columns=feature_names)
    return X_train_proc
