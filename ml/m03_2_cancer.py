from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from operator import mod
from re import T
import numpy as np
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()

# 1. 데이터

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True, random_state=44)

# 1_1. 데이터 전처리

from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. 모델 구성

# model = LinearSVC()
# model.score :  0.9824561403508771
# accuracy_score :  0.9824561403508771

# model = SVC()
# model.score :  0.956140350877193
# accuracy_score :  0.956140350877193

# model = KNeighborsClassifier()
# model.score :  0.956140350877193
# accuracy_score :  0.956140350877193

# model = LogisticRegression()
# model.score :  0.9824561403508771
# accuracy_score :  0.9824561403508771

# model = DecisionTreeClassifier()
# model.score :  0.9385964912280702
# accuracy_score :  0.9385964912280702

model = RandomForestClassifier()
# model.score :  0.9649122807017544
# accuracy_score :  0.9649122807017544

# 3. 컴파일, 훈련

model.fit(x_train, y_train)

# 평가, 예측

from sklearn.metrics import r2_score, accuracy_score

results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("model.score : ",results)
print("accuracy_score : ",acc)

