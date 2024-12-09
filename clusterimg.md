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

# 시각화 & 에러
에러 분석

에러 메시지 silence[(0,0,0)]과 비슷한 문제가 발생했다면, 아래와 같은 원인일 가능성이 높습니다. 특히, PCA를 사용한 시각화 코드에서 자주 발생할 수 있는 문제들을 다뤄보겠습니다.

1. 주요 에러 원인

1.1 클러스터 레이블(cluster_labels) 문제

	•	문제: cluster_labels가 정의되지 않았거나, 올바른 크기를 가지지 않을 때.
	•	cluster_labels는 KMeans 등의 군집화 모델로부터 생성된 클러스터 할당 결과입니다.
	•	cluster_labels의 길이가 X_pca와 일치하지 않으면 에러가 발생합니다.
	•	해결 방법:
	•	cluster_labels는 군집화 모델의 결과로 생성된 배열이어야 하며, 데이터 X와 동일한 크기를 가져야 합니다.

# 클러스터 레이블 확인
print(cluster_labels.shape, X_pca.shape)

1.2 입력 데이터 X 문제

	•	문제: X가 비어있거나, 적절한 형식(2D 배열)이 아닐 때.
	•	PCA는 2D 입력 데이터를 필요로 합니다. 데이터가 1D이거나 누락된 값이 있는 경우 에러 발생.
	•	해결 방법:
	•	X의 크기와 데이터 유형 확인.

print(X.shape)

1.3 PCA로 변환된 데이터 X_pca 문제

	•	문제: X_pca가 생성되지 않거나, 2D 배열이 아닌 경우.
	•	PCA는 n_components=2로 설정 시, 2D 데이터(X_pca[:, 0], X_pca[:, 1])가 생성됩니다.
	•	해결 방법:
	•	X_pca 생성 여부와 크기를 확인.

print(X_pca.shape)

2. 시각화 시 변수 변경 필요

2.1 데이터를 불러올 경우

	•	데이터를 불러왔을 때, 기존 변수명과 새 변수명이 다르면 코드에서 수정해야 합니다.
	•	예: 불러온 데이터의 열 이름 또는 변수를 기반으로 변경.

# 데이터 로드 후 열 이름 확인
import pandas as pd

data = pd.read_csv('your_dataset.csv')
print(data.columns)

2.2 시각화에 필요한 변수 확인

	1.	PCA 입력 데이터 (X):
	•	X는 군집화할 데이터를 포함하며, 범주형 데이터는 제거하거나 인코딩해야 함.
	•	스케일링도 필요할 수 있음.

from sklearn.preprocessing import StandardScaler

# 수치형 데이터만 선택 및 스케일링
X = data.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
X = scaler.fit_transform(X)

	2.	클러스터 레이블 (cluster_labels):
	•	클러스터링 모델로부터 생성된 결과.
	•	예: cluster_labels = kmeans.fit_predict(X).

from sklearn.cluster import KMeans

# 클러스터링 수행
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

3. 수정된 PCA 시각화 코드

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 데이터 로드 및 스케일링
data = pd.read_csv('your_dataset.csv')
X = data.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# 3. PCA를 사용한 2D 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title("Clustering Visualization with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()

4. 최종 점검 사항

	•	데이터 크기와 일치: X, X_pca, cluster_labels 크기가 맞는지 확인.
	•	스케일링: PCA와 클러스터링에 적합한 입력 데이터로 변환했는지 점검.
	•	클러스터 레이블: 군집화 모델로부터 올바르게 생성했는지 확인.

추가 팁

	1.	에러 추적:
	•	KeyError, ValueError 등 에러 메시지가 명확하지 않을 경우, 각 변수의 크기와 내용을 확인.
	2.	변수 변경 시:
	•	불러온 데이터셋의 열 이름이 변경되었다면, 코드에서 이를 반영하여 수정해야 합니다.
	•	예: 데이터셋에 따라 feature1, feature2 등의 이름을 실제 열 이름으로 바꿔야 함.
	3.	의미 있는 열 선택:
	•	PCA와 군집화를 적용할 데이터는 의미 있는 수치형 열만 포함해야 함. 범주형 열은 제거하거나 인코딩.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN은 밀도 기반 군집화 알고리즘으로, 데이터 포인트를 **밀도가 높은 영역(Cluster)**과 **노이즈(Noise)**로 구분합니다.

1. DBSCAN의 특징

	1.	군집의 모양:
	•	비구조적인 클러스터를 탐지(원형이 아닌 모양 가능).
	•	데이터가 불균형하거나 클러스터 크기가 다양한 경우에도 적합.
	2.	하이퍼파라미터:
	•	eps: 두 데이터 포인트가 같은 클러스터로 간주되는 최대 거리.
	•	min_samples: 한 클러스터의 최소 데이터 포인트 수(밀도의 기준).
	3.	노이즈 처리:
	•	밀도가 낮은 점(노이즈)을 클러스터에서 제외.
	4.	초기 클러스터 개수 설정 불필요:
	•	K-means와 달리, 초기 클러스터 개수를 지정하지 않아도 됩니다.

2. DBSCAN 알고리즘의 단계

	1.	각 포인트의 **이웃 거리(eps)**를 확인.
	2.	min_samples 이상의 이웃을 가지는 포인트는 **코어 포인트(Core Point)**로 간주.
	3.	코어 포인트 주변에 있는 포인트를 하나의 클러스터로 병합.
	4.	클러스터에 속하지 못하는 포인트는 **노이즈(Noise)**로 간주.

3. DBSCAN 구현

3.1 데이터 로드 및 전처리

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 데이터 로드
file_path = 'your_dataset.csv'
data = pd.read_csv(file_path)

# 수치형 변수만 선택
numeric_features = data.select_dtypes(include=['int64', 'float64'])

# 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(numeric_features)

3.2 DBSCAN 적용

from sklearn.cluster import DBSCAN
import numpy as np

# DBSCAN 모델 생성
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 클러스터링 수행
cluster_labels = dbscan.fit_predict(X)

# 결과 출력
print(f"Cluster Labels: {np.unique(cluster_labels)}")

3.3 결과 시각화

import matplotlib.pyplot as plt

# 2D 데이터로 변환 (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title("DBSCAN Clustering Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()

4. 하이퍼파라미터 튜닝

DBSCAN은 **eps**와 **min_samples**에 매우 민감합니다. 최적의 파라미터를 찾기 위해 다양한 값을 실험하거나, 이웃 거리의 평균을 사용하여 적절한 값을 찾습니다.

4.1 eps 최적화 (k-거리 그래프)

	•	k-거리 그래프는 가장 가까운 k개의 포인트와의 거리를 계산하여 eps 값을 설정하는 데 도움을 줍니다.

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# k-거리 계산
k = 5  # min_samples와 동일하게 설정
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# 거리 정렬 후 그래프
distances = np.sort(distances[:, k-1], axis=0)
plt.figure(figsize=(8, 6))
plt.plot(distances)
plt.title("k-distance Graph")
plt.xlabel("Data Points Sorted by Distance")
plt.ylabel(f"{k}-distance")
plt.show()

	•	그래프에서 급격히 증가하는 지점을 찾아 eps로 설정.

4.2 min_samples 최적화

	•	min_samples는 일반적으로 데이터 차원(d)에 따라 설정:
￼

5. 결과 해석

	•	클러스터 레이블:
	•	-1: 노이즈(밀도가 낮은 점).
	•	0, 1, 2, ...: 클러스터 번호.

# 클러스터 결과 확인
import numpy as np
unique_clusters = np.unique(cluster_labels)
print(f"Number of clusters (excluding noise): {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
print(f"Number of noise points: {sum(cluster_labels == -1)}")

6. DBSCAN 시각화 및 변수 수정

	•	데이터 불러오기 및 변수 변경:
	•	PCA를 적용한 경우, 새로운 데이터셋을 불러왔을 때 X와 cluster_labels를 다시 생성해야 합니다.

# 데이터 로드
new_data = pd.read_csv('new_dataset.csv')

# 수치형 변수 선택 및 스케일링
numeric_features = new_data.select_dtypes(include=['float64', 'int64'])
X_new = scaler.fit_transform(numeric_features)

# DBSCAN 클러스터링
cluster_labels_new = dbscan.fit_predict(X_new)

# PCA 시각화
X_pca_new = pca.fit_transform(X_new)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_new[:, 0], X_pca_new[:, 1], c=cluster_labels_new, cmap='viridis', alpha=0.7)
plt.title("DBSCAN Visualization (New Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()

7. DBSCAN의 장점과 한계

장점

	•	클러스터의 모양이 다양해도 군집화 가능(비구조적 클러스터 탐지).
	•	군집 개수를 미리 지정할 필요 없음.
	•	노이즈(이상치) 탐지가 가능.

한계

	•	eps와 min_samples 설정에 민감.
	•	클러스터 밀도가 크게 다르면 성능 저하.
	•	데이터 차원이 높을수록 계산 비용 증가.

8. DBSCAN 요약

	1.	전처리:
	•	수치형 데이터 선택 및 스케일링 필수.
	2.	모델링:
	•	DBSCAN의 eps와 min_samples 하이퍼파라미터 설정.
	3.	결과 시각화:
	•	PCA를 사용하여 2D로 변환 후 군집화 결과 확인.
	4.	튜닝:
	•	k-거리 그래프를 활용하여 최적의 eps 값 탐색.


