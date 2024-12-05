# 필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

# 와인 데이터 로드
wine = load_wine()
X = wine.data
y = wine.target

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PCA 수행 (차원 2개로 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# LDA 수행 (차원 2개로 축소)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)

# PCA 결과 차트로 표시
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('PCA of Wine Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

# LDA 결과 차트로 표시
plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_train, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('LDA od Wine Dataset')
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.colorbar()
plt.show()
