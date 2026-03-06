import pandas as pd
import numpy as np

df = pd.read_csv('data/dataset.csv')
df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

x = df[["Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Question Papers Practiced",
        "Extracurricular Activities"]].values

y = df["Performance Index"].values
x = np.c_[np.ones(x.shape[0]), x]
theta = np.linalg.inv(x.T @ x) @ x.T @ y

def predict_performance(hours_studied, prev_scores, sleep, papers, extra):
    extra = 1 if extra == "Yes" else 0
    input_data = np.array([1, hours_studied, prev_scores, sleep, papers, extra])
    prediction = input_data @ theta
    return round(prediction, 2)

def grade(performance_index):
    if performance_index >= 90:
        return "Excellent"
    elif performance_index >= 80:
        return "Good"
    elif performance_index >= 70:
        return "Average"
    elif performance_index >= 60:
        return "Below Average"
    elif performance_index >= 50:
        return "Need Improvement"
    elif performance_index >= 40:
        return "Poor"
    elif performance_index >= 30:
        return "very Poor"
    else:
        return "Fail"

def get_model_performance():
    y_pred = x @ theta
    
    ss_total = np.sum((y - np.sum(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    
    r2 = 1 - (ss_residual / ss_total)
    return round(r2, 4)
    


