


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


