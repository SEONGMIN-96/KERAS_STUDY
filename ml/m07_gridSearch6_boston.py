from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

n_splits = 6
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 모델 : RandomForestRegressor

parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7 ,10]},
    {'min_samples_leaf' : [3, 5, 7 ,10],'n_jobs' : [-1, 2, 4]},
    {'min_samples_split' : [2, 3, 5, 10],'n_jobs' : [-1, 2, 4]},
    {'n_jobs' : [-1, 2, 4],'n_estimators' : [100, 200]}
]

# model = GridSearchCV(RandomForestClassifier(), parameter, cv=kfold)
model = GridSearchCV(RandomForestRegressor(), parameter, cv=kfold)

model.fit(x, y)

print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)

# 최적의 매개변수 : RandomForestRegressor(max_depth=12)
# best_score_ : 0.8646242829557936