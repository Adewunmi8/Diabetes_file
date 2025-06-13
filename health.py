import pandas as pd
import streamlit as st
import matplotlib.pyplot as pt
import numpy as np
import seaborn as sns

st.title("DIABETES_ANALYSIS")

#import my csv file
st.markdown("# FIRST FIVE")
df = pd.read_csv("diabetes.csv")
st.write(df.head())

st.markdown("# LAST FIVE")
df = pd.read_csv("diabetes.csv")
st.write(df.tail())


