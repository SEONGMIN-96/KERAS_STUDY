from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.datasets import load_wine

# 1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, shuffle=True)

# 1_2. preprocessing

from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. 모델 구성

# model = LinearSVC()
# model.score :  0.8055555555555556
# accuracy_score :  0.8055555555555556

# model = SVC()
# model.score :  0.7222222222222222
# accuracy_score :  0.7222222222222222

# model = KNeighborsClassifier()
# model.score :  0.6944444444444444
# accuracy_score :  0.6944444444444444

# model = LogisticRegression()
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222

# model = DecisionTreeClassifier()
# model.score :  0.9166666666666666
# accuracy_score :  0.9166666666666666

model = RandomForestClassifier()
# model.score :  0.9444444444444444
# accuracy_score :  0.9444444444444444

# 3. 컴파일, 훈련

model.fit(x_train, y_train)

# 4. 평가, 예측

from sklearn.metrics import r2_score, accuracy_score

results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("model.score : ",results)
print("accuracy_score : ",acc)