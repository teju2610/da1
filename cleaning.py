import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the dataset
df = pd.read_csv("titanic-dataset.csv")
print("🔹 Original Dataset Preview:")
print(df.head())

# Step 3: Data Cleaning
print("\n🔹 Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing Age values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' column due to excessive nulls
df.drop(columns=['Cabin'], inplace=True)

print("\n🔹 Missing values after cleaning:")
print(df.isnull().sum())
print("\n🔹 Data integration: Not applicable (single dataset used)")
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
print("\n🔹 Transformed Data Preview:")
print(df[['Sex', 'Embarked']].head())
print("\n🔹 Plotting graphs...")
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.show()
df.to_csv("titanic-cleaned.csv", index=False)
print("\n Preprocessed data saved as 'titanic-cleaned.csv'")