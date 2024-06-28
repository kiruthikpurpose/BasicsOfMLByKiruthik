# Program to demonstrate Principal Component Analysis (PCA) for Dimensionality Reduction

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generating synthetic data
X, y = make_classification(n_samples=100, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementing PCA for dimensionality reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Visualizing data after PCA
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Reduced Dimension Data')
plt.show()

# Applying logistic regression on PCA-transformed data
lr = LogisticRegression()
lr.fit(X_train_pca, y_train)

# Predicting on test data
y_pred = lr.predict(X_test_pca)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after PCA: {accuracy:.2f}")
