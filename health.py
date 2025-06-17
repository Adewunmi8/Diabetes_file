import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly_express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


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

st.markdown("### CORRELATION")
correlation = df.corr()
st.write(correlation)

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

#create a new column. 'blood' == name of new column, where is a method/fumction in numpy

"""
df2["blood"] = np.where(df["Pregnancies"]< 10, "Less than ten weeks", "Greater than ten weeks")
st.write(df2)
counter = df2["blood"].value_counts().reset_index

fig2 = px.bar(df2, x =df["blood"], y =["BloodPressure"], title=("Distribution of Blood Pressure vs Pregnancies Graph"))
st.plotly_chart(fig2, use_container_width=True)
    """
    
    
#PREDICTIVE DATA ANALYSIS(PREDICTION)
st.markdown("## PREDICTIVE ANALYSIS")
#use drop function to remove a particular column from the table 
x = df.drop("Outcome", axis=1) 
Y = df["Outcome"] 

x_train,x_test,Y_train,Y_test = train_test_split(x,Y,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,Y_train)

st.markdown("## Outcome Prediction")
prediction = model.predict(x_test)

st.write(prediction)

st.markdown("## Model Evaluation")
accuracy = accuracy_score(prediction, Y_test)
st.write(accuracy)









