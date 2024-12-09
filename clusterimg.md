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
