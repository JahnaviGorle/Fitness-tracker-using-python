import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import base64
import io
import json
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #FF4B4B;
        font-weight: 600;
        padding-top: 1rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .highlight {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #FF4B4B;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
    }
    .stat-container {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .progress-label {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .feedback-box {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 0.5rem solid #4CAF50;
    }
    .download-box {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 0.5rem solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">Personal Fitness Tracker</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div class="highlight">
    Track and predict calories burned based on your physical attributes and exercise metrics. 
    Adjust the parameters in the sidebar to see how different factors affect your calorie burn.
    </div>
    """, unsafe_allow_html=True)

# Sidebar styling and inputs
st.sidebar.markdown('<div class="sidebar-header">User Input Parameters</div>', unsafe_allow_html=True)

def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30, help="Your age in years")
    st.sidebar.markdown("---")
    
    bmi = st.sidebar.slider("BMI:", 15.0, 40.0, 20.0, 0.1, help="Body Mass Index")
    st.sidebar.markdown("---")
    
    duration = st.sidebar.slider("Exercise Duration (minutes):", 0, 35, 15, help="How long you exercised")
    st.sidebar.markdown("---")
    
    heart_rate = st.sidebar.slider("Heart Rate (bpm):", 60, 130, 80, help="Your average heart rate during exercise")
    st.sidebar.markdown("---")
    
    body_temp = st.sidebar.slider("Body Temperature (°C):", 36.0, 42.0, 38.0, 0.1, help="Your body temperature during exercise")
    st.sidebar.markdown("---")
    
    gender_button = st.sidebar.radio("Gender:", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# Load and preprocess data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    return calories, exercise

calories, exercise = load_data()

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
@st.cache_data
def train_model():
    random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    random_reg.fit(X_train, y_train)
    return random_reg

random_reg = train_model()

# Align prediction data columns with training data
df_model = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df_model)

# Display user parameters
st.markdown('<div class="sub-header">Your Parameters</div>', unsafe_allow_html=True)

with st.expander("View Your Input Parameters", expanded=True):
    # Create a more visually appealing parameter display
    cols = st.columns(3)
    with cols[0]:
        st.metric("Age", f"{df['Age'].values[0]} years")
        st.metric("Gender", "Male" if df['Gender_male'].values[0] == 1 else "Female")
    
    with cols[1]:
        st.metric("BMI", f"{df['BMI'].values[0]}")
        st.metric("Exercise Duration", f"{df['Duration'].values[0]} min")
    
    with cols[2]:
        st.metric("Heart Rate", f"{df['Heart_Rate'].values[0]} bpm")
        st.metric("Body Temperature", f"{df['Body_Temp'].values[0]} °C")

# Prediction section with visual emphasis
st.markdown('<div class="sub-header">Calorie Prediction</div>', unsafe_allow_html=True)

# Create columns for centered prediction display
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown('<div class="stat-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-value">{round(prediction[0], 2)}</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.5rem;">kilocalories</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Similar results section
st.markdown('<div class="sub-header">Similar Results</div>', unsafe_allow_html=True)

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]

if not similar_data.empty:
    with st.expander("View Similar Cases", expanded=True):
        st.dataframe(similar_data.sample(min(5, len(similar_data))).style.highlight_max(axis=0))
else:
    st.info("No similar cases found in the dataset.")

# General information with visual meters
st.markdown('<div class="sub-header">How You Compare</div>', unsafe_allow_html=True)

# Calculate percentile statistics
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

age_pct = round(sum(boolean_age) / len(boolean_age), 2) * 100
duration_pct = round(sum(boolean_duration) / len(boolean_duration), 2) * 100
heart_rate_pct = round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100
body_temp_pct = round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100

# Create a visual representation of the percentiles
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="progress-label">Age Percentile</div>', unsafe_allow_html=True)
    st.progress(age_pct/100)
    st.caption(f"You are older than {age_pct}% of other people.")
    
    st.markdown('<div class="progress-label">Exercise Duration Percentile</div>', unsafe_allow_html=True)
    st.progress(duration_pct/100)
    st.caption(f"Your exercise duration is higher than {duration_pct}% of other people.")

with col2:
    st.markdown('<div class="progress-label">Heart Rate Percentile</div>', unsafe_allow_html=True)
    st.progress(heart_rate_pct/100)
    st.caption(f"You have a higher heart rate than {heart_rate_pct}% of other people during exercise.")
    
    st.markdown('<div class="progress-label">Body Temperature Percentile</div>', unsafe_allow_html=True)
    st.progress(body_temp_pct/100)
    st.caption(f"You have a higher body temperature than {body_temp_pct}% of other people during exercise.")

# NEW: Feedback Section
st.markdown('<div class="sub-header">Provide Feedback</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
    st.write("We value your feedback! Please let us know how accurate you feel this prediction is.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        accuracy_rating = st.slider("Accuracy Rating", 1, 5, 3, help="How accurate do you think this prediction is?")
    
    with col2:
        if accuracy_rating <= 2:
            st.warning("We're sorry the prediction wasn't accurate for you. Please leave a comment to help us improve.")
        elif accuracy_rating >= 4:
            st.success("Thank you for the positive feedback!")
        else:
            st.info("Thank you for your feedback.")
    
    feedback_text = st.text_area("Additional Comments", height=100, help="Please provide any additional feedback or suggestions.")
    
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button("Submit Feedback", use_container_width=True):
            # Here you would typically save the feedback to a database
            st.session_state.feedback_submitted = True
            st.success("Feedback submitted successfully! Thank you for helping us improve.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# NEW: Download Options
st.markdown('<div class="sub-header">Download Your Results</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    st.write("Save your fitness data and predictions for future reference.")
    
    # Create a dictionary of all data to be downloaded
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    download_data = {
        "timestamp": timestamp,
        "user_parameters": df.to_dict(orient="records")[0],
        "prediction": {
            "calories_burned": float(round(prediction[0], 2))
        },
        "percentile_data": {
            "age_percentile": float(age_pct),
            "duration_percentile": float(duration_pct),
            "heart_rate_percentile": float(heart_rate_pct),
            "body_temperature_percentile": float(body_temp_pct)
        }
    }
    
    # Function to download data as JSON
    def get_json_download_link(data):
        json_str = json.dumps(data, indent=4)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="fitness_data_{timestamp.replace(" ", "_").replace(":", "-")}.json" class="download-button">Download JSON</a>'
        return href
    
    # Function to download data as CSV
    def get_csv_download_link(data):
        # Convert the nested dictionary to a flat dataframe
        flat_data = pd.json_normalize(data)
        csv = flat_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="fitness_data_{timestamp.replace(" ", "_").replace(":", "-")}.csv" class="download-button">Download CSV</a>'
        return href
    
    # Generate a PDF report
    def get_pdf_report():
        buffer = io.BytesIO()
        # In a real app, you'd create a PDF here
        pdf_content = f"Fitness Tracker Report\n\nDate: {timestamp}\n\nUser Parameters:\n"
        for k, v in download_data["user_parameters"].items():
            pdf_content += f"- {k}: {v}\n"
        pdf_content += f"\nPredicted Calories Burned: {download_data['prediction']['calories_burned']}\n"
        
        # Convert string to bytes
        buffer.write(pdf_content.encode())
        buffer.seek(0)
        
        # Create download link
        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="fitness_report_{timestamp.replace(" ", "_").replace(":", "-")}.txt" class="download-button">Download Report</a>'
        return href
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(get_json_download_link(download_data), unsafe_allow_html=True)
        st.caption("Download as JSON")
    
    with col2:
        st.markdown(get_csv_download_link(download_data), unsafe_allow_html=True)
        st.caption("Download as CSV")
    
    with col3:
        st.markdown(get_pdf_report(), unsafe_allow_html=True)
        st.caption("Download as Report")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Option to save to history
if st.checkbox("Save this session to your fitness history"):
    exercise_name = st.text_input("Exercise Name (optional)", "Workout Session")
    
    if st.button("Save to History"):
        # Here you would typically save to a database
        st.success(f"Saved '{exercise_name}' to your fitness history!")
        st.info("You can access your saved sessions in the History tab.")

# Footer
st.markdown("---")
st.caption("Personal Fitness Tracker - Powered by Machine Learning")