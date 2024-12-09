군집화(Clustering)는 비지도 학습의 한 유형으로, 데이터를 그룹으로 나누는 작업입니다. 데이터 전처리와 시각화 방법을 자세히 설명하며, 학습한 데이터를 시각화할 때 필요한 변경 사항도 함께 다룹니다.

1. 데이터 로드 및 탐색

1.1 데이터 로드

군집화를 수행할 데이터셋을 로드합니다.

import pandas as pd

# 데이터 로드
file_path = 'your_dataset.csv'
data = pd.read_csv(file_path)

# 데이터의 구조 확인
print(data.head())  # 데이터의 처음 5행
print(data.info())  # 데이터 타입 및 결측값 확인
print(data.describe())  # 수치형 열의 통계 요약

1.2 데이터 시각화

군집화 전에 데이터 분포를 이해하기 위해 기본 시각화를 수행합니다.

import matplotlib.pyplot as plt
import seaborn as sns

# 예: 두 수치형 열 간의 관계
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='feature1', y='feature2', alpha=0.7)
plt.title('Feature1 vs Feature2')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

2. 데이터 전처리

2.1 결측값 처리

군집화 모델은 결측값을 처리할 수 없으므로, 결측값을 제거하거나 적절히 대체해야 합니다.

# 결측값 확인
print(data.isnull().sum())

# 결측값 처리: 수치형은 중앙값, 범주형은 최빈값으로 대체
data['numeric_column'].fillna(data['numeric_column'].median(), inplace=True)
data['categorical_column'].fillna(data['categorical_column'].mode()[0], inplace=True)

2.2 범주형 데이터 처리

범주형 데이터를 숫자로 변환해야 모델이 학습할 수 있습니다.

# 원-핫 인코딩
data = pd.get_dummies(data, columns=['categorical_column'], drop_first=True)

2.3 스케일링

군집화 알고리즘(K-means, DBSCAN 등)은 데이터의 거리 계산에 민감하므로, 모든 수치형 데이터를 스케일링해야 합니다.

from sklearn.preprocessing import StandardScaler

# 수치형 열 선택 및 스케일링
numeric_cols = ['feature1', 'feature2', 'feature3']
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

3. 군집화 수행

3.1 K-Means 클러스터링

K-Means 알고리즘을 사용하여 데이터를 군집화합니다.

from sklearn.cluster import KMeans

# 클러스터 개수 설정 (예: 3개)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(data)

# 결과 저장
data['Cluster'] = y_kmeans

3.2 최적의 클러스터 개수 찾기

1) 엘보우 방법

군집화 결과를 평가하기 위해 WCSS(Within-Cluster Sum of Squares)를 계산합니다.

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

2) 실루엣 점수

군집화의 품질을 평가합니다.

from sklearn.metrics import silhouette_score

silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

4. 결과 시각화

4.1 2D PCA 시각화

고차원 데이터를 2D로 축소하여 클러스터링 결과를 시각화합니다.

from sklearn.decomposition import PCA

# PCA로 2D 데이터로 축소
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Clustering Visualization with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

4.2 원래 데이터의 특정 열을 사용한 시각화

특정 열을 사용해 클러스터 분포를 시각화할 수 있습니다.

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='feature1', y='feature2', hue='Cluster', palette='viridis', alpha=0.7)
plt.title('Clustering Result')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend(title='Cluster')
plt.show()

5. 학습된 데이터 파일로 시각화하려면 바꿀 부분

5.1 데이터 로드

학습된 데이터 파일을 불러와 군집화 결과를 사용하여 시각화합니다.

# 군집화 결과 포함된 데이터 로드
file_path = 'clustered_dataset.csv'
clustered_data = pd.read_csv(file_path)

# PCA를 사용한 시각화
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_pca = pca.fit_transform(clustered_data.drop('Cluster', axis=1))  # Cluster 열 제외

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clustered_data['Cluster'], cmap='viridis', alpha=0.7)
plt.title('Clustering Visualization with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

5.2 주요 수정 포인트

	1.	군집화 결과를 포함한 열(Cluster)이 데이터에 포함되었는지 확인:
	•	포함되지 않은 경우, 데이터 전처리 후 다시 군집화를 수행해야 합니다.
	2.	Cluster 열을 제외한 나머지 열로 PCA 시각화:
	•	Cluster는 타겟 변수로 간주되므로 PCA 계산 시 제외합니다.
	3.	군집 결과를 색상으로 시각화:
	•	sns.scatterplot이나 plt.scatter에서 c 또는 hue에 Cluster 열을 사용.

요약

	1.	데이터 전처리:
	•	결측값 처리, 범주형 데이터 처리, 스케일링 적용.
	2.	군집화 수행:
	•	K-Means, DBSCAN 등 사용.
	•	엘보우 방법과 실루엣 점수로 최적의 클러스터 개수 선택.
	3.	결과 시각화:
	•	PCA로 2D 축소 후 시각화.
	•	원래 데이터의 주요 열을 사용한 클러스터링 시각화.
	4.	학습된 데이터 시각화:
	•	Cluster 열을 기준으로 시각화.
	•	PCA나 특정 열을 활용하여 결과를 직관적으로 표현.

이 프로세스를 통해 군집화와 결과 시각화를 효과적으로 수행할 수 있습니다!