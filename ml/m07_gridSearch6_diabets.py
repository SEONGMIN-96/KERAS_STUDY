import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings
from sklearn.metrics import accuracy_score
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

# 모델 : RandomForestRegressor
parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7 ,10]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7 ,10]},
    {'min_samples_leaf' : [3, 5, 7 ,10],'n_jobs' : [-1, 2, 4]},
    {'min_samples_split' : [2, 3, 5, 10],'n_jobs' : [-1, 2, 4]},
    {'n_jobs' : [-1, 2, 4],'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7 ,10]}
]

start_time = time.time()
# model = GridSearchCV(RandomForestClassifier(), parameter, cv=kfold)
model = GridSearchCV(RandomForestRegressor(), parameter, cv=kfold, verbose=1)

model.fit(x_train, y_train)

print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)
print("model.score :", model.score(x_test, y_test))
print("소요 시간 :", time.time() - start_time)

# 최적의 매개변수 : RandomForestRegressor(max_depth=12, min_samples_leaf=10)
# best_score_ : 0.9530690631769104
# model.score : 0.9358486320006627
# 소요 시간 : 109.6659517288208