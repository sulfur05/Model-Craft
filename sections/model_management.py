from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import joblib
import pandas as pd
import streamlit as st


EXPORT_ROOT = Path("exports")
BUNDLE_FILENAME = "model_bundle.joblib"
METADATA_FILENAME = "metadata.json"

REQUIRED_BUNDLE_KEYS = {
    "model",
    "preprocessor",
    "feature_columns",
    "target_column",
    "task_type",
    "trained_model_name",
    "created_at",
    "version",
    "dataset_shape",
    "metrics",
    "library_versions",
}


def _safe_name(name: str) -> str:
    """Make a filesystem-safe model name."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "model"


def _timestamp_version() -> str:
    """Create a sortable version string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _library_versions() -> Dict[str, str]:
    """Record the main library versions used to create the bundle."""
    versions: Dict[str, str] = {}
    for name, module in {
        "streamlit": st,
        "pandas": pd,
        "joblib": joblib,
    }.items():
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def _ensure_export_dir(model_name: str, version: str, root: Union[str, Path] = EXPORT_ROOT) -> Path:
    """Create and return the export directory for a given model/version."""
    export_dir = Path(root) / _safe_name(model_name) / version
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def build_model_bundle(
    model_name: Optional[str] = None,
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete export bundle from Streamlit session state.

    This is the single source of truth for export.
    It packages the trained estimator, preprocessing pipeline, schema, and metadata.
    """
    required_session_keys = ["trained_model", "preprocessor", "feature_columns"]
    missing = [key for key in required_session_keys if key not in st.session_state]
    if missing:
        raise ValueError(
            "Missing required session_state values: " + ", ".join(missing)
        )

    model = st.session_state["trained_model"]
    preprocessor = st.session_state["preprocessor"]
    feature_columns = st.session_state["feature_columns"]
    target_column = st.session_state.get("target_column")
    task_type = st.session_state.get("task_type", "classification")
    trained_model_name = model_name or st.session_state.get("trained_model_name", type(model).__name__)
    created_at = datetime.now().isoformat(timespec="seconds")
    bundle_version = version or _timestamp_version()

    metrics = st.session_state.get("trained_model_metrics", {})
    comparison_results = st.session_state.get("model_comparison_results")
    params = None
    if hasattr(model, "get_params"):
        try:
            params = model.get_params()
        except Exception:
            params = None

    dataset = st.session_state.get("dataset")
    dataset_shape = tuple(dataset.shape) if hasattr(dataset, "shape") else None

    bundle: Dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "task_type": task_type,
        "trained_model_name": trained_model_name,
        "created_at": created_at,
        "version": bundle_version,
        "dataset_shape": dataset_shape,
        "metrics": metrics,
        "comparison_results": comparison_results,
        "params": params,
        "library_versions": _library_versions(),
    }

    return bundle


def validate_model_bundle(bundle: Dict[str, Any]) -> None:
    """Validate the bundle structure before save/load/use."""
    if not isinstance(bundle, dict):
        raise TypeError("Model bundle must be a dictionary.")

    missing = sorted(REQUIRED_BUNDLE_KEYS - set(bundle.keys()))
    if missing:
        raise ValueError("Bundle is missing required keys: " + ", ".join(missing))

    if not isinstance(bundle["feature_columns"], (list, tuple)):
        raise TypeError("bundle['feature_columns'] must be a list or tuple.")

    if not isinstance(bundle["trained_model_name"], str):
        raise TypeError("bundle['trained_model_name'] must be a string.")