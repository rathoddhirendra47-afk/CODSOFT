# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Libraries import successfully")

# Load dataset
df = pd.read_csv("train.csv")
print(df.head())

#Understand the Dataset
print(df.info())
print(df.describe())

#Check Missing values
print(df.isnull().sum())

#Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True)
print(df.isnull().sum())

#Survival Distribution Visualization
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.show()
print("Survival Distribution Chart")

#Survival by Gender Visualization
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()
print("Gender-wise Survival Chart")

#Age Distribution Visualization
plt.hist(df['Age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
print("Age Distribution Chart ")

#Encode Categorical Variables
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

print(df.head())

#Feature Selection
X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df['Survived']

print(X.head())
print(y.head())

#Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()
print("Heatmap")

#Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

#Train the Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
print("Model Trained Successfully")

#Make Predictions
y_pred = model.predict(X_test)
print(y_pred[:10])

#Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy =", accuracy)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Visualize Confusion Matrix
sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#Feature Importance Analysis
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)

print("="*70)
print("        TITANIC SURVIVAL PREDICTION PROJECT COMPLETED")
print("="*70)

print("✓ Libraries Imported Successfully")
print("✓ Dataset Loaded Successfully")
print("✓ Dataset Overview Completed")
print("✓ Missing Values Handled")
print("✓ Data Cleaning Completed")
print("✓ Exploratory Data Analysis Completed")
print("✓ Data Visualization Completed")
print("✓ Feature Selection Completed")
print("✓ Dataset Split Successfully")
print("✓ Random Forest Model Trained Successfully")
print("✓ Predictions Generated")
print("✓ Model Evaluation Completed")
print("✓ Confusion Matrix Generated")
print("✓ Accuracy Score Calculated")

print("\nProject Status : SUCCESS")
print("Task 1 Completed Successfully!")

print("="*70)
print("="*70)