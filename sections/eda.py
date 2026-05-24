import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MAX_EDA_ROWS = 10_000

#---------
def _top_categorical_info(series: pd.Series) -> (str, float, int):
    vc = series.value_counts(dropna=False)
    if vc.empty:
        return "no values", 0.0, 0
    top = vc.index[0]
    freq = int(vc.iloc[0])
    share = float(freq / series.size)
    unique = series.nunique(dropna=True)
    return str(top), share, int(unique)

def dataset_eda(df: pd.DataFrame, numeric_cols, categorical_cols):

    if len(df) > MAX_EDA_ROWS:
        df_sample = df.sample(MAX_EDA_ROWS, random_state=42)

        st.caption(
            f"showing EDA on a random sample of {MAX_EDA_ROWS} rows "
            f"out of {len(df)} to keep things responsive"
        )
    else:
        df_sample = df

    st.subheader("Summary")
    st.write(f"- Rows: {df.shape[0]}")
    st.write(f"- Columns: {df.shape[1]}")


    #below code will work on giving numeric summary
    if numeric_cols:
        st.markdown("**Numeric columns summary**")
        st.dataframe(df[numeric_cols].describe().T)
    else:
        st.write("No numeric columns detected.")


    #categorical columns summary
    if categorical_cols:
        st.markdown("**Categorical column value counts**")
        cat_col = st.selectbox(
            "Choose a categorical column to see its most common values",
            options=categorical_cols,
        )
        vc = df[cat_col].value_counts(dropna=False).head(20)
        st.write(vc)
        top, share, unique = _top_categorical_info(df[cat_col])

    



def dataset_not_available():
    st.info("Upload a dataset in step 1 (Dataset Upload) first.")
    return

def eda_section():
    with st.expander("2. Exploratory Data Analysis(EDA)"):
        st.expander("Explore your data")

        if "dataset" not in st.session_state:
            dataset_not_available()
            return
        
        df = st.session_state["dataset"]
        numeric_cols = st.session_state.get("numeric_columns", [])

        categorical_cols = st.session_state.get("categorical_columns", [])

        st.write(
            "Click the button below to generate summary statistics and visualisations "
            "for your dataset."
        )

        run_eda = st.button("Run EDA")

        if not run_eda:
            return
        
        dataset_eda(df, numeric_cols, categorical_cols)


        
        