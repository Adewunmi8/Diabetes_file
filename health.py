import pandas as pd
import streamlit as st
import matplotlib.pyplot as pt
import numpy as np
import seaborn as sns
import plotly_express as px

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

st.markdown("### Blood Pressure")
st.write(df["BloodPressure"].describe()) 

st.markdown("### FIRST FIVE BLOOD PRESSURE")
st.write(df["BloodPressure"].head())

#UNIVARIATE ANALYSIS
st.markdown("## UNIVARIATE ANALYSIS")

st.markdown("### Blood Pressure")
st.write(df["BloodPressure"].describe())


fig = px.bar(df["BloodPressure"], y= "BloodPressure", title="Distribution of Blood Pressure Graph")
st.plotly_chart(fig, use_container_width=True)


#BIVARIATE ANALYSIS
st.markdown("## BIVARIATE ANALYSIS")

st.markdown("### Blood Pressure vs Pregnancies")
df2 = pd.DataFrame(df["Pregnancies"], df["BloodPressure"])
st.write(df2)

fig2 = px.bar(df, x ='Pregnancies', y ='BloodPressure', title=("Distribution of Blood Pressure vs Pregnancies Graph"))
st.plotly_chart(fig2, use_container_width=True)