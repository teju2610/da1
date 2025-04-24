import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))





import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('/content/knn.csv')
data['Study Hours'] = pd.to_numeric(data['Study Hours'], errors='coerce')
data['Exam Score'] = pd.to_numeric(data['Exam Score'], errors='coerce')
data['Class'] = pd.to_numeric(data['Class'], errors='coerce')
data.dropna(inplace=True)
X = data[['Study Hours', 'Exam Score']].values
y = data['Class'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
new_data_point = np.array([[3.8, 68]])
new_data_point_scaled = scaler.transform(new_data_point)
knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_euclidean.fit(X_scaled, y)
result_euclidean = knn_euclidean.predict(new_data_point_scaled)
knn_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn_manhattan.fit(X_scaled, y)
result_manhattan = knn_manhattan.predict(new_data_point_scaled)
print(f"Class (using Euclidean distance): {'Pass' if result_euclidean[0] == 1 else 'Fail'}")
print(f"Class (using Manhattan distance): {'Pass' if result_manhattan[0] == 1 else 'Fail'}")
