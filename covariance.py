***********mean vector,covariance,correlation******************
import numpy as np
import pandas as pd
data = {
    'X1': [2, 4, 6, 8, 10],
    'X2': [1, 3, 5, 7, 9],
    'X3': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)
print("Original Dataset:\n", df)
mean_vector = df.mean()
print("\nMean Vector:\n", mean_vector)
variance = df.var(ddof=1)  
print("\nVariance:\n", variance)
covariance_matrix = df.cov()
print("\nCovariance Matrix:\n", covariance_matrix)
correlation_matrix = df.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)