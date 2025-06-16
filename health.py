import pandas as pd
import streamlit as st
import matplotlib.pyplot as pt
import numpy as np
import seaborn as sns

st.title("# *DIABETES_ANALYSIS*")
st.markdown("## OVERVIEW")

#import my csv file
st.markdown("### FIRST FIVE")
df = pd.read_csv("diabetes.csv")
st.write(df.head())

st.markdown("### LAST FIVE")
df = pd.read_csv("diabetes.csv")
st.write(df.tail())

st.markdown("### DATA INFO")
my = df.info()
st.write(my)

st.markdown("### DATA DESCRIBE")
my = df.describe()
st.write(my)

st.markdown("### DATA SHAPE")
my = df.shape
st.write(my)

#UNIVARIATE ANALYSIS
st.markdown("## UNIVARIATE ANALYSIS")

st.markdown("### Blood Pressure")
st.write(df["BloodPressure"].describe())