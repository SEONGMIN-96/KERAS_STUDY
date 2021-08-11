import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# 1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

# train_test_split -> k_fold

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True, random_state=44)

n_splits = 6
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

paramiters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"]},
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

# 모델
# GridSearchCV = model x paramiter x cv 만큼의 실행 
# GridSearchCV 를통해 알아낸 최적의 값을 SVC에 대입해 같은 결과값을 뽑아내는지 확인

# model = GridSearchCV(SVC(), paramiters, cv=kfold)
model = SVC(C=1000, kernel='linear')

# 훈련

model.fit(x_train, y_train)

# 평가, 예측

# print("최적의 매개변수 :", model.best_estimator_)
# print("best_score_ :", model.best_score_)

y_predict = model.predict(x_test)

print("model.score :", model.score(x_test, y_test))
print("acc :", accuracy_score(y_test, y_predict))