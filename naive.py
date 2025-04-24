**************Gaussian Naive Bayes (using Iris dataset)*******************

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("=== Gaussian Naive Bayes ===")
print(classification_report(y_test, y_pred))

***************Multinomial Naive Bayes (using text data)*******************

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample text data
texts = ['I love programming', 'Python is great', 'I hate bugs', 'Debugging is hard', 'I love Python']
labels = [1, 1, 0, 0, 1]  # 1: Positive, 0: Negative

# Multinomial Naive Bayes Pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Predict
preds = model.predict(['I love debugging', 'Bugs are annoying'])
print("\n=== Multinomial Naive Bayes ===")
print("Predictions:", preds)

****************Bernoulli Naive Bayes (binary features)***********************
from sklearn.naive_bayes import BernoulliNB
import numpy as np

# Binary feature data (e.g., word presence)
X = np.array([
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 1]
])
y = [1, 1, 0, 0]  # Labels

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X, y)
y_pred = bnb.predict([[1, 0, 0, 1]])
print("\n=== Bernoulli Naive Bayes ===")
print("Prediction:", y_pred)




//NaiveByes


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv('/content/naiveBayes.csv')

# Features and target
X = df[['Free', 'Win', 'Money', 'Offer']].values
y = df['Spam'].values

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# New data point (e.g., Free=1, Win=0, Money=0, Offer=0)
new_data_point = [[1, 0, 0, 0]]
predicted_class = model.predict(new_data_point)[0]

# Display result
predicted_label = 'Spam' if predicted_class == 1 else 'Not Spam'
print(f'Predicted class for new data point {new_data_point[0]}: {predicted_label}')

//multinomialNB

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
messages = pd.read_csv('/content/Multinomial_Naive_Bayes.csv', encoding='latin-1')

# Clean column names
messages.columns = messages.columns.str.strip()

# Check required columns
if 'Message' not in messages.columns or 'Label' not in messages.columns:
    raise KeyError("CSV must have columns 'Message' and 'Label'")

# Vectorize text messages (convert to bag-of-words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages['Message'])

# Target labels
y = messages['Label']

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Classify a new message
test_message = ["Congratulations, lunch at 5PM"]
X_test = vectorizer.transform(test_message)

# Predict
predicted_label = model.predict(X_test)[0]
probs = model.predict_proba(X_test)[0]

# Output results
print(f"Message: {test_message[0]}")
print(f"P(Spam | Message) = {probs[list(model.classes_).index('Spam')]}")
print(f"P(Not Spam | Message) = {probs[list(model.classes_).index('Not Spam')]}")
print(f"Predicted Class: {predicted_label}")


//gaussian

import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv('/content/Gaussian_NB(dataset).csv')

# Check column names
print(data.columns)

# Extract features and target
X = data[['Study Hours', 'Sleep Hours']]
y = data['Class(pass=1,fail=0)']  # Ensure correct column name

# Create and train the model
model = GaussianNB()
model.fit(X, y)

# New data point to predict
new_data = [[4.3, 3.2]]

# Make prediction
predicted_class = model.predict(new_data)[0]
probs = model.predict_proba(new_data)[0]

# Output results
print(f"Predicted Class: {'Pass' if predicted_class == 1 else 'Fail'}")
print(f"P(Fail | x) = {probs[0]}")
print(f"P(Pass | x) = {probs[1]}")


//bernoulli NB

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelBinarizer

# Load dataset
data = pd.read_csv('/content/Bernoulli_NB(dataset).csv')
data.columns = data.columns.str.strip()

# Features and target
features = ["Give birth", "Can fly", "Live in water", "Have legs"]
X_raw = data[features]
y_raw = data["Class"].str.strip().str.lower()  # Target labels

# Convert 'yes'/'no' to binary 1/0
X = X_raw.applymap(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

# Encode class labels: 'mammal' -> 1, 'non-mammal' -> 0
y = (y_raw == "mammal").astype(int)

# Train Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(X, y)

# New test sample
test_sample = {"Give birth": "yes", "Can fly": "no", "Live in water": "no", "Have legs": "no"}
test_input = [[1 if test_sample[feature].strip().lower() == "yes" else 0 for feature in features]]

# Prediction
predicted_class = model.predict(test_input)[0]
probabilities = model.predict_proba(test_input)[0]

# Output
print(f"\nTest Sample: {test_sample}")
print(f"P(Mammal | X) = {probabilities[1]}")
print(f"P(Non-Mammal | X) = {probabilities[0]}")
print(f"Predicted Class: {'mammal' if predicted_class == 1 else 'non-mammal'}\n")