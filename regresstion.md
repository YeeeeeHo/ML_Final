# classification 방법
데이터셋을 처음 받아 분류(Classification) 모델을 수행하기 위한 전체 프로세스를 아래 단계별로 자세히 설명합니다. 각각의 단계는 데이터 특성과 목표에 따라 조정할 수 있습니다.

1. 데이터 로드 및 탐색

1.1 데이터 로드

	•	CSV 파일 등을 로드하여 pandas 데이터프레임으로 저장.

import pandas as pd

# 데이터 로드
file_path = 'your_dataset.csv'
data = pd.read_csv(file_path)

# 데이터의 구조 확인
print(data.head())  # 처음 5행 확인
print(data.info())  # 열 정보 및 데이터 타입 확인
print(data.describe())  # 수치형 열의 통계 요약

1.2 결측값 확인

	•	결측값을 처리해야 데이터의 품질을 보장할 수 있습니다.

# 결측값 확인
print(data.isnull().sum())

1.3 타겟 변수 확인

	•	타겟 변수(분류 대상)가 **이진(0/1)**인지 다중 클래스인지 확인.
	•	타겟 변수의 분포를 확인하여 데이터가 불균형한지 체크.

# 타겟 변수 분포
target = 'target_column_name'  # 타겟 변수명
print(data[target].value_counts())

2. 데이터 전처리

2.1 결측값 처리

	•	결측값이 있는 열이나 행을 처리.
	•	수치형 데이터: 평균, 중앙값으로 대체.
	•	범주형 데이터: 최빈값으로 대체.

# 수치형 결측값 대체 (중앙값)
data['numeric_column'].fillna(data['numeric_column'].median(), inplace=True)

# 범주형 결측값 대체 (최빈값)
data['categorical_column'].fillna(data['categorical_column'].mode()[0], inplace=True)

2.2 불필요한 열 제거

	•	모델에 의미가 없거나, 타겟 변수 예측에 직접적으로 연관된 열(예: ID, 이름)을 제거.

# 불필요한 열 삭제
data.drop(['unnecessary_column1', 'unnecessary_column2'], axis=1, inplace=True)

2.3 범주형 데이터 처리

	•	라벨 인코딩(Label Encoding): 순서가 있는 범주형 데이터.
	•	원-핫 인코딩(One-Hot Encoding): 순서가 없는 범주형 데이터.

from sklearn.preprocessing import LabelEncoder

# 라벨 인코딩
label_encoder = LabelEncoder()
data['encoded_column'] = label_encoder.fit_transform(data['categorical_column'])

# 원-핫 인코딩
data = pd.get_dummies(data, columns=['categorical_column'], drop_first=True)

2.4 스케일링

	•	필요 이유: 대부분의 머신러닝 모델은 특성 간의 값 범위가 크면 성능이 저하될 수 있음.
	•	적용 대상: 수치형 데이터.

from sklearn.preprocessing import StandardScaler

# 수치형 열 스케일링
scaler = StandardScaler()
data[['numeric_column1', 'numeric_column2']] = scaler.fit_transform(data[['numeric_column1', 'numeric_column2']])

3. 데이터 분리

	•	데이터를 **훈련(Train)**과 테스트(Test) 세트로 나눕니다.

from sklearn.model_selection import train_test_split

# 타겟 변수와 독립 변수 분리
X = data.drop('target_column_name', axis=1)  # 특성
y = data['target_column_name']  # 타겟 변수

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

4. 분류 모델 구축

4.1 모델 선택

	•	주요 분류 모델:
	•	로지스틱 회귀 (Logistic Regression)
	•	의사결정 나무 (Decision Tree)
	•	랜덤 포레스트 (Random Forest)
	•	서포트 벡터 머신 (SVM)
	•	XGBoost, LightGBM

4.2 모델 학습

from sklearn.ensemble import RandomForestClassifier

# 모델 생성 및 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

4.3 모델 평가

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 예측
y_pred = model.predict(X_test)

# 평가
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

4.4 중요 변수 확인 (특성 중요도)

# 특성 중요도
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
features = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

5. 모델 튜닝

5.1 하이퍼파라미터 튜닝

	•	GridSearchCV 또는 RandomizedSearchCV를 사용하여 모델 성능 최적화.

from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 설정
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV 실행
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

6. 결과 시각화

	•	Confusion Matrix 시각화:

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

7. 결과 분석 및 결론

	•	모델 성능이 적절한지 확인 (예: 정확도, F1 점수 등).
	•	타겟 변수의 불균형 여부 확인.
	•	모델이 과적합되었는지 테스트 세트 성능을 통해 판단.

요약

	1.	데이터 로드 및 탐색: 데이터 구조와 결측값 파악.
	2.	데이터 전처리: 결측값 처리, 범주형 데이터 인코딩, 스케일링.
	3.	데이터 분리: 훈련 데이터와 테스트 데이터 분리.
	4.	모델 학습 및 평가: 모델 학습, 예측, 성능 평가.
	5.	모델 튜닝: 하이퍼파라미터 최적화로 성능 향상.
	6.	시각화 및 분석: 혼동 행렬과 특성 중요도 시각화.

## 데이터 전처리 스케일링 및 범주형 데이터

데이터 전처리에서 범주형 데이터 처리와 스케일링은 서로 다른 목적과 데이터를 대상으로 합니다. 각각 어떤 데이터에 대해 적용해야 하는지 예시와 함께 설명하겠습니다.

1. 범주형 데이터 처리

적용 대상

	•	범주형 데이터(Categorical Data): 값이 특정 범주(카테고리)를 나타내는 데이터.
	•	데이터가 텍스트형이거나, 숫자형이라도 범주적 의미를 가진다면 처리해야 합니다.
	•	예: Gender, Color, Region 등.

처리 방법

1) 라벨 인코딩 (Label Encoding)

	•	범주형 데이터가 순서가 있는 경우 사용.
	•	문자열 데이터를 정수로 변환.
	•	예: ['Low', 'Medium', 'High'] → [0, 1, 2].

2) 원-핫 인코딩 (One-Hot Encoding)

	•	범주형 데이터가 순서가 없는 경우 사용.
	•	각 범주를 새로운 열로 만들어 이진값(0 또는 1)으로 표시.
	•	예: ['Red', 'Green', 'Blue'] →

Red   Green   Blue
 1      0       0
 0      1       0
 0      0       1

예시

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 데이터 예시
data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Female', 'Male'],  # 순서 없음
    'Education': ['High School', 'Bachelor', 'Master', 'PhD']  # 순서 있음
})

# 라벨 인코딩 (Education)
label_encoder = LabelEncoder()
data['Education_encoded'] = label_encoder.fit_transform(data['Education'])

# 원-핫 인코딩 (Gender)
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)

print(data)

결과

Education	Gender	Education_encoded	Gender_Male
High School	Male	1	1
Bachelor	Female	0	0
Master	Female	2	0
PhD	Male	3	1

처리 대상과 비처리 대상

데이터 열	처리 여부	이유
Gender	원-핫 인코딩 필요	순서가 없고 범주형 데이터.
Education	라벨 인코딩 필요	순서가 있는 범주형 데이터.
Age	처리 불필요	수치형 데이터.
Income	처리 불필요	수치형 데이터.

2. 스케일링

적용 대상

	•	수치형 데이터(Numerical Data): 연속적인 값을 가지며, 값의 크기가 의미 있는 데이터.
	•	스케일링은 머신러닝 모델에서 특성 간 크기 차이가 모델 성능에 영향을 줄 때 필요.
	•	예: Age, Income, Height, Weight 등.

처리 방법

1) 표준화 (Standardization)

	•	데이터를 평균 0, 표준편차 1로 변환.
	•	사용 모델: SVM, KNN, PCA 등 거리 기반 알고리즘.
	•	수식:
￼

2) 정규화 (Normalization)

	•	데이터를 0~1 범위로 변환.
	•	사용 모델: 신경망, 로지스틱 회귀 등.
	•	수식:
￼

예시

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 데이터 예시
data = pd.DataFrame({
    'Age': [25, 30, 35, 40],
    'Income': [50000, 60000, 75000, 100000]
})

# 표준화
scaler = StandardScaler()
data[['Age_standardized', 'Income_standardized']] = scaler.fit_transform(data[['Age', 'Income']])

# 정규화
normalizer = MinMaxScaler()
data[['Age_normalized', 'Income_normalized']] = normalizer.fit_transform(data[['Age', 'Income']])

print(data)

결과

Age	Income	Age_standardized	Income_standardized	Age_normalized	Income_normalized
25	50000	-1.34	-1.07	0.00	0.00
30	60000	-0.45	-0.56	0.25	0.20
35	75000	0.45	0.06	0.50	0.50
40	100000	1.34	1.57	1.00	1.00

처리 대상과 비처리 대상

데이터 열	처리 여부	이유
Age	스케일링 필요	수치형 데이터. 모델 학습에 스케일링 필요.
Income	스케일링 필요	수치형 데이터. 값 범위가 큼.
Gender	처리 불필요	범주형 데이터. 스케일링 대상 아님.
Education	처리 불필요	범주형 데이터. 스케일링 대상 아님.

결론

범주형 데이터 처리

	•	텍스트형 데이터: 반드시 처리 필요.
	•	숫자로 표현된 범주형 데이터: 범주적 의미가 있다면 처리.

스케일링

	•	수치형 데이터에 적용.
	•	특성 값의 범위 차이가 클수록 스케일링이 중요.
	•	거리 기반 알고리즘(SVM, KNN, PCA 등)에서 반드시 필요.


# regresstion
**Regression(회귀)**은 연속적인 숫자 값을 예측하는 작업입니다. 데이터 전처리와 결과 시각화를 포함하여 전체 프로세스를 단계별로 자세히 설명하겠습니다.

1. 데이터 로드 및 탐색

1.1 데이터 로드

데이터를 불러와 구조를 확인합니다.

import pandas as pd

# 데이터 로드
file_path = 'your_dataset.csv'
data = pd.read_csv(file_path)

# 데이터의 첫 5행 확인
print(data.head())

# 데이터 타입과 결측값 확인
print(data.info())

# 수치형 열의 통계 확인
print(data.describe())

1.2 타겟 변수 확인

타겟 변수(예측할 값)를 지정하고, 데이터 분포를 확인합니다.

import matplotlib.pyplot as plt

# 타겟 변수의 분포 시각화
target = 'target_column_name'
plt.hist(data[target], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Target Variable Distribution')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.show()

2. 데이터 전처리

2.1 결측값 처리

결측값이 있는 데이터를 처리합니다.

# 결측값 확인
print(data.isnull().sum())

# 수치형 변수의 결측값을 중앙값으로 대체
data['numeric_column'].fillna(data['numeric_column'].median(), inplace=True)

# 범주형 변수의 결측값을 최빈값으로 대체
data['categorical_column'].fillna(data['categorical_column'].mode()[0], inplace=True)

2.2 불필요한 열 제거

분석에 불필요한 열(예: ID, 이름)을 제거합니다.

data.drop(['unnecessary_column1', 'unnecessary_column2'], axis=1, inplace=True)

2.3 범주형 데이터 처리

범주형 데이터를 숫자로 변환합니다.

1) 라벨 인코딩

	•	순서가 있는 범주형 데이터에 적합.

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['encoded_column'] = label_encoder.fit_transform(data['categorical_column'])

2) 원-핫 인코딩

	•	순서가 없는 범주형 데이터에 적합.

data = pd.get_dummies(data, columns=['categorical_column'], drop_first=True)

2.4 스케일링

수치형 데이터에 대해 스케일링을 적용합니다.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['numeric_column1', 'numeric_column2']] = scaler.fit_transform(data[['numeric_column1', 'numeric_column2']])

2.5 데이터 분리

독립 변수(X)와 종속 변수(y)를 분리하고, 훈련/테스트 데이터를 나눕니다.

from sklearn.model_selection import train_test_split

# 독립 변수와 종속 변수 분리
X = data.drop('target_column_name', axis=1)
y = data['target_column_name']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

3. 회귀 모델 구축

3.1 모델 선택

	•	가장 기본적인 회귀 모델: 선형 회귀(Linear Regression).
	•	복잡한 데이터에서는 랜덤 포레스트, XGBoost 등을 사용할 수 있음.

from sklearn.linear_model import LinearRegression

# 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

3.2 모델 평가

모델의 예측 성능을 평가합니다.

from sklearn.metrics import mean_squared_error, r2_score

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

4. 결과 시각화

4.1 실제 값과 예측 값 비교

# 실제 값과 예측 값 비교
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

4.2 잔차 분석

잔차(실제 값 - 예측 값)를 확인하여 모델의 적합성을 평가합니다.

# 잔차 계산
residuals = y_test - y_pred

# 잔차 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='red')
plt.axhline(y=0, color='blue', linestyle='--', linewidth=2)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

5. 모델 튜닝

복잡한 모델(XGBoost, 랜덤 포레스트 등)의 경우, 하이퍼파라미터 튜닝을 통해 성능을 개선합니다.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# 그리드 검색
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R-squared:", grid_search.best_score_)

**요약

	1.	데이터 로드 및 탐색: 데이터 구조와 분포를 파악.
	2.	데이터 전처리:
	•	결측값 처리, 불필요한 열 제거.
	•	범주형 데이터 처리 (라벨 인코딩, 원-핫 인코딩).
	•	수치형 데이터 스케일링.
	3.	데이터 분리: 훈련/테스트 데이터로 분리.
	4.	모델 학습 및 평가: 회귀 모델 학습, MSE와 R² 평가.
	5.	결과 시각화: 실제 값 vs. 예측 값, 잔차 분석.
	6.	모델 튜닝: 하이퍼파라미터 최적화.


