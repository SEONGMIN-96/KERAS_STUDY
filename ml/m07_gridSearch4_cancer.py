from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

from operator import mod
from re import T
import numpy as np
from sklearn.datasets import load_breast_cancer
import time

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

# 1. 데이터

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True, random_state=44)

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
model = GridSearchCV(RandomForestClassifier(), parameter, cv=kfold)

model.fit(x_train, y_train)
end_time = time.time() - start_time

print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)
print("model.score :", model.score(x_test, y_test))

# 최적의 매개변수 : RandomForestClassifier(max_depth=6)
# best_score_ : 0.9648538011695907
# model.score : 0.9649122807017544