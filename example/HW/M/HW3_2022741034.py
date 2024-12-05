import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일을 로드
file_path = 'hw.csv'
data = pd.read_csv(file_path)

# 데이터 행렬
data.head()

# 특징(feature)과 타겟(target)을 분리
X = data.drop(columns=['target'])
y = data['target']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LDA를 사용하여 2차원으로 차원 축소
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# k-distance plot 생성 (내림차순 정렬)
k = 5  # MinPts 값에 해당하는 k 값 설정
nearest_neighbors = NearestNeighbors(n_neighbors=k)
nearest_neighbors.fit(X_lda)
distances, indices = nearest_neighbors.kneighbors(X_lda)

# k번째 이웃 거리 (k-distances) 오름차순 정렬
distances = np.sort(distances[:, k-1])

# k-distance plot 그리기
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-NN distance")
plt.title("Sorted k-distance Graph for Determining Optimal eps")
plt.grid()
plt.show()

# Kneedle 알고리즘을 사용하여 최적의 eps 값 찾기
kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve="convex", direction="increasing")
best_eps = distances[kneedle.elbow]
print(f"자동으로 감지된 최적의 eps 값: {best_eps}")

# DBSCAN Clustering with optimal eps
dbscan = DBSCAN(eps=best_eps, min_samples=4)
dbscan_labels = dbscan.fit_predict(X_lda)

# Clustering with other algorithms

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_lda)

# Mean Shift Clustering
bandwidth = estimate_bandwidth(X_lda, quantile=0.2)
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift_labels = mean_shift.fit_predict(X_lda)

# Gaussian Mixture Model (GMM) Clustering
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X_lda)

# 시각화
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].scatter(X_lda[:, 0], X_lda[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
axs[0, 0].set_title("K-Means Clustering")
axs[0, 1].scatter(X_lda[:, 0], X_lda[:, 1], c=mean_shift_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
axs[0, 1].set_title("Mean Shift Clustering")
axs[1, 0].scatter(X_lda[:, 0], X_lda[:, 1], c=gmm_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
axs[1, 0].set_title("GMM Clustering")
axs[1, 1].scatter(X_lda[:, 0], X_lda[:, 1], c=dbscan_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
axs[1, 1].set_title(f"DBSCAN Clustering (eps={best_eps}, MinPts={4})")

plt.tight_layout()
plt.show()
