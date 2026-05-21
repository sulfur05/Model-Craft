from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from utils.reporting import create_pdf_report_bytes
import numpy as np

from sections.model_management import (
    load_model_bundle,
    list_saved_models,
    predict_with_bundle,
    save_model_bundle,
    validate_model_bundle,
    build_model_bundle,
)

from utils.reporting import create_pdf_report_bytes

def _display_bundle_summary(bundle: dict) -> None:
    model_name = bundle.get("trained_model_name", "Unknown")
    version = bundle.get("version", "Unknown")
    task_type = bundle.get("task_type", "Unknown")
    target_column = bundle.get("target_column", "Unknown")
    created_at = bundle.get("created_at", "Unknown")
    dataset_shape = bundle.get("dataset_shape", "Unknown")
    metrics = bundle.get("metrics", {})

    st.write(f"**Model:** {model_name}")
    st.write(f"**Version:** {version}")
    st.write(f"**Task type:** {task_type}")
    st.write(f"**Target column:** {target_column}")
    st.write(f"**Created at:** {created_at}")
    st.write(f"**Dataset shape:** {dataset_shape}")

    if metrics:
        st.subheader("Saved metrics")
        st.json(metrics)

    if bundle.get("comparison_results") is not None:
        st.subheader("Model comparison results")
        try:
            st.dataframe(bundle["comparison_results"])
        except Exception:
            st.write(bundle["comparison_results"])