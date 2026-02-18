import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import predict_performance, grade, get_model_performance
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

df = pd.read_csv('studentPerformance.csv')
df["Grade"] = df["Performance Index"].apply(grade)

st.title("STUDENT PERFORMANCE ANALYZER🧑‍🎓💯")
st.write("This app predicts student performance based on acedamic details and activities.")

st.sidebar.subheader("Academic Details")

Hours_Studied = st.sidebar.slider("Hours Studied", 1, 9, 0)
prev_scores = st.sidebar.slider("Previous scores", 40, 100, 0)
sleep = st.sidebar.slider("Sleep Hours", 4, 9, 0)
papers = st.sidebar.slider("Sample Questions Practiced", 0, 9, 0)

st.sidebar.subheader("Activities")
extra = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])

if st.button("Predict Performance"):
    
    st.subheader("Student performance Chart")
    result = predict_performance(Hours_Studied, prev_scores, sleep, papers, extra)
    student_grade = grade(result)
    
    st.subheader("Predicted Results")
    st.write(f"Predicted Performance Index: {result}")
    st.write(f"Grade: {student_grade}")


if st.button("Model Performance"): 
    st.subheader("Model Performance")
    st.write("R-squared Score:", get_model_performance())

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

