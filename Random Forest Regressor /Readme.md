🏋️ Calorie Burner Prediction – Kaggle Playground Series S5E5

This project was created as part of the Kaggle competition Playground Series - Season 5, Episode 5, which focused on predicting calorie expenditure during physical exercise based on physiological and activity-related features.

🔍 Competition Objective

The goal was to predict the number of calories burned (Calories) based on various input features such as age, height, weight, heart rate, and exercise type.

Evaluation Metric:
Root Mean Squared Logarithmic Error (RMSLE)

🧠 Modeling Approach

I used a Random Forest Regressor to build a robust and interpretable regression model.

Model: RandomForestRegressor
Score (RMSLE): 0.06032 on Kaggle's public leaderboard

calorie-burner.ipynb: Jupyter Notebook containing all code, preprocessing, model training, and evaluation steps
train.csv: Training dataset provided by Kaggle
test.csv: Test dataset provided by Kaggle


📊 Feature Description

Feature	Description
Age	Age of the individual (in years)
Height	Height (in cm)
Weight	Weight (in kg)
Duration	Duration of the activity (in minutes)
Heart_Rate	Average heart rate
Body_Temp	Body temperature (°C)
BMI	Body Mass Index
Gender	Gender (Male/Female)
Exercise	Type of exercise (categorical)

📈 Model Performance

Final RMSLE: 0.06032
Public Leaderboard: Result submitted on Kaggle
Interpretation:
The RMSLE of 0.06032 suggests the model predicts calorie burn with a very low relative logarithmic error — solid performance among competition participants.