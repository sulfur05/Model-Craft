import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MAX_EDA_ROWS = 10_000

def dataset_not_available():
    st.info("Upload a dataset in step 1 (Dataset Upload) first.")
    return

def eda_section():
    with st.expander("2. Exploratory Data Analysis(EDA)"):
        st.expander("Explore your data")

        if "dataset" not in st.session_state:
            dataset_not_available()
            return
        
        