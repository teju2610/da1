import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Sample dataset
data = {
    'X1': [2, 4, 6, 8, 10],
    'X2': [1, 3, 5, 7, 9],
    'X3': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)
print("Original Data:\n", df)

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Step 3: Apply PCA
pca = PCA()
pca_components = pca.fit_transform(X_scaled)

# Step 4: Display results
print("\nPCA Components (transformed data):\n", pd.DataFrame(pca_components, columns=['PC1', 'PC2', 'PC3']))
print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)