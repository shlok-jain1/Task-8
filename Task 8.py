#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

STUDENT_DATA_FILE = "C:\\Users\\Admin\\Downloads\\student_scores_with_target.xlsx"
FRAUD_DATA_FILE = "C:\\Users\\Admin\\Downloads\\fraud_detection_dataset_v2.xlsx"

# Section 1: Student Performance Prediction
print("--- Section 1: Student Performance Prediction ---")

student_data = pd.read_excel(STUDENT_DATA_FILE)
student_data["Pass/Fail"] = LabelEncoder().fit_transform(student_data["Pass/Fail"])

scaler_student_baseline = StandardScaler()
numerical_features_student_baseline = ["Attendance Rate (%)", "Math Score", "Science Score", "English Score", "History Score", "Geography Score"]
student_data[numerical_features_student_baseline] = scaler_student_baseline.fit_transform(student_data[numerical_features_student_baseline])

X_student_baseline = student_data[numerical_features_student_baseline]
y_student_baseline = student_data["Pass/Fail"]

X_train_student_baseline, X_test_student_baseline, y_train_student_baseline, y_test_student_baseline = train_test_split(X_student_baseline, y_student_baseline, test_size=0.2, random_state=42, shuffle=True)

baseline_model_student = DecisionTreeClassifier(random_state=42)
baseline_model_student.fit(X_train_student_baseline, y_train_student_baseline)
y_pred_baseline_student = baseline_model_student.predict(X_test_student_baseline)
print("Comparison of Student Data Models:")
print("\nBaseline Model Performance:\n", classification_report(y_test_student_baseline, y_pred_baseline_student))

student_data["Total Score"] = student_data[["Math Score", "Science Score", "English Score", "History Score", "Geography Score"]].sum(axis=1)

scaler_student = StandardScaler()
numerical_features_student = ["Total Score", "Attendance Rate (%)", "Math Score", "Science Score", "English Score", "History Score", "Geography Score"]
student_data[numerical_features_student] = scaler_student.fit_transform(student_data[numerical_features_student])

X_student = student_data[numerical_features_student]
y_student = student_data["Pass/Fail"]

X_train_student, X_test_student, y_train_student, y_test_student = train_test_split(X_student, y_student, test_size=0.2, random_state=42, shuffle=True)

param_grid_student = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

grid_search_student = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_student, cv=5, scoring='accuracy')
grid_search_student.fit(X_train_student, y_train_student)

best_model_student = grid_search_student.best_estimator_

y_pred_student = best_model_student.predict(X_test_student)
print("\nBest Model Performance (with Total Score and GridSearchCV):\n", classification_report(y_test_student, y_pred_student))

# Section 2: Fraud Detection with Decision Trees
print("\n--- Section 2: Fraud Detection with Decision Trees ---")

fraud_data = pd.read_excel(FRAUD_DATA_FILE)
fraud_data['Amount'] = fraud_data['Amount'].fillna(fraud_data['Amount'].mean())
fraud_data['Transaction Method'] = fraud_data['Transaction Method'].fillna(fraud_data['Transaction Method'].mode()[0])

categorical_features_fraud = ['Type', 'Location', 'Transaction Method']
for feature in categorical_features_fraud:
    le_fraud = LabelEncoder()
    fraud_data[feature] = le_fraud.fit_transform(fraud_data[feature])

fraud_data['Amount'] = np.log1p(fraud_data['Amount'])

features_fraud = ['Amount', 'Type', 'Customer ID', 'Merchant ID', 'Location', 'Transaction Method', 'Account Balance']
X_fraud = fraud_data[features_fraud]
y_fraud = fraud_data['Is Fraud']

X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, shuffle=True)

model_fraud = DecisionTreeClassifier(random_state=42)
model_fraud.fit(X_train_fraud, y_train_fraud)

y_pred_fraud = model_fraud.predict(X_test_fraud)
print("\nFraud Data - Model Performance:\n", classification_report(y_test_fraud, y_pred_fraud))

print("\nFraud Data - Feature Importance:")
importances_fraud = model_fraud.feature_importances_
for feature, importance in zip(X_fraud.columns, importances_fraud):
    print(f"{feature}: {importance:.4f}")

print("\nFraud Detection Improvement Recommendations:")
print("- Address class imbalance using techniques like SMOTE or class weighting.")
print("- Explore additional features such as time-based or frequency-based features.")
print("- Consider using more advanced models like Random Forest or Gradient Boosting Machines.")
print("- Perform hyperparameter tuning to optimize model parameters.")


# In[ ]:




