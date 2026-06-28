#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Libraries Imported Successfully")

#Load Dataset
df = pd.read_csv("creditcard.csv")
print(df.head())

#Dataset Information
print("Dataset Shape:", df.shape)
print(df.info())

# Check Missing Values
print(df.isnull().sum())
print(df.describe())

#Class Distribution Analysis
print(df['Class'].value_counts())

#Fraud vs Genuine Transactions visualization
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)

plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()

#Correlation Heatmap visualization
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Matrix")

plt.show()

#Normalize Amount Feature
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

#Feature and Target Separation
X = df.drop("Class", axis=1)
y = df["Class"]

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)

#Train Logistic Regression Model
model = LogisticRegression(
    solver='liblinear',
    max_iter=5000,
    random_state=42
)

model.fit(X_train, y_train)
print("✓ Model Training  Successfully")

#Prediction
y_pred = model.predict(X_test)
print("✓ Predictions Generated")

#Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy =", round(accuracy*100,2), "%")

#Classification Report
print(classification_report(y_test,y_pred))

#Confusion Matrix Visualization
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

#Transaction Amount Distribution Visualization
plt.figure(figsize=(8,5))
sns.histplot(df['Amount'],
             bins=50,
             kde=True)

plt.title("Transaction Amount Distribution")
plt.show()

#Fraud vs Normal Transaction Amount Visualization
plt.figure(figsize=(8,5))
sns.boxplot(x='Class',
            y='Amount',
            data=df)

plt.title("Fraud vs Normal Transaction Amount")
plt.show()

print("="*70)
print("        CREDIT CARD FRAUD DETECTION PROJECT COMPLETED")
print("="*70)

print("✓ Libraries Imported Successfully")
print("✓ Dataset Loaded Successfully")
print("✓ Data Cleaning Completed")
print("✓ Missing Values Checked")
print("✓ Data Visualization Completed")
print("✓ Data Normalization Completed")
print("✓ Dataset Split Successfully")
print("✓ Fraud Detection Model Trained Successfully")
print("✓ Fraud Predictions Generated")
print("✓ Model Evaluation Completed")
print("✓ Precision, Recall and F1-Score Calculated")
print("✓ Confusion Matrix Generated")

print("\nProject Status : SUCCESS")
print("Task 5 Completed Successfully!")

print("="*70)
print("="*70)