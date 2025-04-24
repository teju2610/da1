*****************Binary Logistic Regression (2 classes)***************************

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load binary classification dataset
data = load_breast_cancer()
X = data.data
y = data.target  # Binary: 0 = malignant, 1 = benign

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Binary Logistic Regression
model_bin = LogisticRegression()
model_bin.fit(X_train, y_train)
y_pred_bin = model_bin.predict(X_test)

print("=== Binary Logistic Regression ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bin))
print("Classification Report:\n", classification_report(y_test, y_pred_bin))

******************Multinomial Logistic Regression (more than 2 classes)**************************

from sklearn.datasets import load_iris

# Load Iris dataset (3 classes)
iris = load_iris()
X = iris.data
y = iris.target  # Multiclass: 0, 1, 2

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Multinomial Logistic Regression
model_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model_multi.fit(X_train, y_train)
y_pred_multi = model_multi.predict(X_test)

print("\n=== Multinomial Logistic Regression ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_multi))
print("Classification Report:\n", classification_report(y_test, y_pred_multi))




//Binary Logistic

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("data.csv")  # Replace with your actual file name

# Selecting features and target
X = df[['Age', 'Salary']]
y = df['Purchased']  # Assuming 'Purchased' is the target column

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Printing the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

//multinomial Logistic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data.csv")  # Replace with your actual file name

# Feature and target selection
X = df[['Age', 'Salary']]  # Replace with actual column names
y = df['Education level']  # Target variable with more than 2 classes

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf)
print("Classification Report:\n", class_report)