# Hotel Booking Cancellation Prediction 🏨📊

This project simulates a scenario in the sales department of a hotel business, where the goal is to predict which customers are most likely to cancel their bookings. We use machine learning techniques, such as **One-Hot Encoding**, **Binarization**, and **RandomForestClassifier**, to analyze and predict customer booking behavior.

**Source Data:**  
- `hotel_bookings_training.csv` – Raw hotel booking data used for training the model.  
- `new_customers.csv` – Data of new customers for which cancellation predictions are made.  

**Project Files:**  
- `main.py` – Python script containing the code for data preprocessing, model training, and prediction.  
- `inference_pipeline.joblib` – Saved model used for predicting cancellations.  
- `requirements.txt` – Required dependencies for the project.  
- `README.md` – Documentation for the project.  

## Features

- **Data Preprocessing:**  
  - Removes personal customer information to maintain privacy.  
  - Applies One-Hot Encoding to categorical variables to convert them into a format suitable for machine learning.  
  - Binarizes certain variables to convert them into binary values, ensuring consistency across the dataset.  

- **Data Transformation:**  
  - Scales numerical features using **RobustScaler** to handle outliers and improve model performance.  
  - Keeps some features without transformation to maintain original data characteristics.  

- **Model Training:**  
  - Trains a RandomForestClassifier to predict the likelihood of a booking being canceled.  
  - Evaluates the model using accuracy and recall metrics to measure its performance.  
  - Saves the trained model as a `.joblib` file for later use.  

- **Prediction:**  
  - Makes predictions on new customers using the trained model.  
  - Predicts the probability of cancellation for each customer, helping the business target high-risk clients.

## Project Structure

