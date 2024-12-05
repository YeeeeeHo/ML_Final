# 필요한 라이브러리 임포트
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 데이터 불러오기
data = load_breast_cancer()
X = data.data
y = data.target

# 학습/테스트 데이터로 분리 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정 트리 분류기 생성 및 학습
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 트리 구조 시각화
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names, rounded=True)
plt.show()
