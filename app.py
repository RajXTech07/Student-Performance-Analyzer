import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import predict_performance, grade, get_model_performance
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

df = pd.read_csv('data/dataset.csv')
df["Grade"] = df["Performance Index"].apply(grade)

st.set_page_config(page_title="Student Performance Analyzer", layout="wide")

st.title("🎓 Student Performance Analyzer")
st.markdown(
"""
Predict a student's **Performance Index** using a Linear Regression model built from scratch.

Fill in the student details below and click **Predict Performance**.
"""
)


st.subheader("📝 Enter Student Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        hours = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
        prev_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)

    with col2:
        sleep = st.number_input("Sleep Hours", min_value=0, max_value=12, value=6)
        papers = st.number_input("Sample Papers Practiced", min_value=0, max_value=20, value=3)

    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    predict_button = st.form_submit_button("Predict Performance 🚀")
    
# Prediction Result Section
if predict_button:

    result = predict_performance(hours, prev_scores, sleep, papers, extra)
    student_grade = grade(result)

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Performance Index", result)

    with col2:
        st.metric("Grade", student_grade)

# Plotting Graph using Plotly
if st.button("Show Animated Grade Chart"):

    st.subheader("📊 Animated Grade Distribution")

    grade_counts = df["Grade"].value_counts().reset_index()
    grade_counts.columns = ["Grade", "Count"]

    fig = px.bar(
        grade_counts,
        x="Grade",
        y="Count",
        color="Grade",
        title="Student Grade Distribution",
        text="Count",
    )

    fig.update_traces(textposition='outside')

    fig.update_layout(
        xaxis_title="Grade",
        yaxis_title="Number of Students",
        showlegend=False,
        template="plotly_dark"  # try removing this for light theme
    )

    st.plotly_chart(fig, use_container_width=True)


# for model performance...
if st.button("Model Performance"): 
    st.subheader("Model Performance")
    st.write("Performance Score:", get_model_performance())