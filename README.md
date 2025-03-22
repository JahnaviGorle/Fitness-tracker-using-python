# Fitness-tracker-using-python
A Streamlit-based web application that predicts the number of kilocalories burned based on user input parameters like age, BMI, exercise duration, heart rate, and body temperature.
Features 🚀
✔️ User-friendly interface using Streamlit
✔️ Predicts calories burned using Random Forest Regression
✔️ Takes user inputs such as Age, BMI, Exercise Duration, Heart Rate, and Body Temperature
✔️ Compares user stats with other similar data points
✔️ Displays similar calorie-burning cases for reference

Tech Stack 🛠
Frontend: Streamlit

Backend: Python

Machine Learning Models: Random Forest Regressor

Libraries Used: pandas, numpy, matplotlib, seaborn, sklearn

Installation & Usage
1️⃣ Clone the repository:
git clone https://github.com/your-username/personal-fitness-tracker.git
cd personal-fitness-tracker
2️⃣ Install dependencies
3️⃣ Run the application:streamlit run app.py
4️⃣ Upload the required dataset files (calories.csv & exercise.csv)
Dataset 📊
The model uses exercise and calorie data from the calories.csv and exercise.csv files. The dataset includes:

User_ID

Gender

Age

Height & Weight

Duration of Exercise

Heart Rate & Body Temperature

Calories Burned

Model & Prediction 🔍
The application trains a Random Forest Regressor model with the preprocessed dataset.
📌 How it works:

Takes user inputs

Preprocesses the data to match model features

Predicts calorie burn using Random Forest Regression

Displays similar results and comparison statistics
