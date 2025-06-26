import streamlit as st
import pandas as pd
import os

@st.cache_data
def cargar_raw() -> pd.DataFrame:
    return pd.read_parquet("./clean_data/semi_raw.parquet")

@st.cache_data
def cargar_process_robot() -> pd.DataFrame:
    return pd.read_parquet("./clean_data/process_robot.parquet")

@st.cache_data
def cargar_avg_global() -> pd.DataFrame:
    return pd.read_parquet("./clean_data/avg_global.parquet")