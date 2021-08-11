from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import time

import numpy as np
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

# 1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, shuffle=True)

n_splits = 6
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 모델 : RandomForestClassifier

parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7 ,10]},
    {'min_samples_leaf' : [3, 5, 7 ,10],'n_jobs' : [-1, 2, 4]},
    {'min_samples_split' : [2, 3, 5, 10],'n_jobs' : [-1, 2, 4]},
    {'n_jobs' : [-1, 2, 4],'n_estimators' : [100, 200]}
]

start_time = time.time()

# model = GridSearchCV(RandomForestClassifier(), parameter, cv=kfold, verbose=1)
# Fitting 6 folds for each of 54 candidates, totalling 324 fits
# 최적의 매개변수 : RandomForestClassifier(max_depth=6, n_estimators=200)
# best_score_ : 0.9886973180076629
# 소요 시간 : 63.45900630950928

model = RandomizedSearchCV(RandomForestClassifier(), parameter, cv=kfold, verbose=1)
# Fitting 6 folds for each of 10 candidates, totalling 60 fits
# 최적의 매개변수 : RandomForestClassifier(min_samples_leaf=3, n_jobs=-1)
# best_score_ : 0.9886973180076629
# 소요 시간 : 12.759056091308594

model.fit(x, y)

print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)
print("소요 시간 :", time.time() - start_time)