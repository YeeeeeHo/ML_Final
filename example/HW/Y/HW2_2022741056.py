# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target
target_names = wine.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (necessary for LDA)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# PCA: Reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# LDA: Reduce dimensions to 2
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)

# Plotting results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# PCA plot
for i, target_name in enumerate(target_names):
    ax1.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
ax1.set_title('PCA of Wine Dataset')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend()

# LDA plot
for i, target_name in enumerate(target_names):
    ax2.scatter(X_lda[y_train == i, 0], X_lda[y_train == i, 1], label=target_name)
ax2.set_title('LDA of Wine Dataset')
ax2.set_xlabel('LD1')
ax2.set_ylabel('LD2')
ax2.legend()

# Save the plot
plt.savefig('wine_pca_lda_plot.png')
plt.show()
