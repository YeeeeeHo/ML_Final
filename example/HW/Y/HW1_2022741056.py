# 필요한 라이브러리 임포트
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# 1. 유방암 데이터 불러오기
data = load_breast_cancer()
X = data.data  # 특성 데이터
y = data.target  # 타겟 레이블 (0 = 양성, 1 = 악성)

# 2. 학습/테스트 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 결정 트리 모델 학습
clf = DecisionTreeClassifier(random_state=42)  # 결정 트리 모델 생성
clf.fit(X_train, y_train)  # 학습

# 4. 트리 구조 시각화
plt.figure(figsize=(20,10))  # 그림 크기 설정
tree.plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()

# 5. 테스트 데이터로 예측하고 성능 평가
accuracy = clf.score(X_test, y_test)
print(f"테스트 데이터에서의 정확도: {accuracy:.4f}")
