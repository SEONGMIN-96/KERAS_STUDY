from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations
import warnings
import time

warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline

n_splits = 6
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 모델 : RandomForestRegressor

# parameter = [
#     {'randomforestregressor__n_estimators' : [100, 200], 'randomforestregressor__max_depth' : [6, 8, 10, 12]},
#     {'randomforestregressor__max_depth' : [6, 8, 10, 12],'randomforestregressor__min_samples_leaf' : [3, 5, 7 ,10]},
#     {'randomforestregressor__min_samples_leaf' : [3, 5, 7 ,10],'randomforestregressor__n_jobs' : [-1, 2, 4]},
#     {'randomforestregressor__min_samples_split' : [2, 3, 5, 10],'randomforestregressor__n_jobs' : [-1, 2, 4]},
#     {'randomforestregressor__n_jobs' : [-1, 2, 4],'randomforestregressor__n_estimators' : [100, 200]}
# ]

parameter = [
    {'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10, 12]},
    {'rf__max_depth' : [6, 8, 10, 12],'rf__min_samples_leaf' : [3, 5, 7 ,10]},
    {'rf__min_samples_leaf' : [3, 5, 7 ,10],'rf__n_jobs' : [-1, 2, 4]},
    {'rf__min_samples_split' : [2, 3, 5, 10],'rf__n_jobs' : [-1, 2, 4]},
    {'rf__n_jobs' : [-1, 2, 4],'rf__n_estimators' : [100, 200]}
]

start_time = time.time()

# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
pipe = Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestRegressor())])

# model = GridSearchCV(pipe, parameter, cv=kfold, verbose=1)
# Fitting 6 folds for each of 54 candidates, totalling 324 fits
# 최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('randomforestregressor',
#                  RandomForestRegressor(min_samples_split=10, n_jobs=2))])
# best_score_ : 0.8533605143795571
# 소요 시간 : 75.72138953208923

model = RandomizedSearchCV(pipe, parameter, cv=kfold, verbose=1)
# Fitting 6 folds for each of 10 candidates, totalling 60 fits
# 최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('randomforestregressor',
#                  RandomForestRegressor(min_samples_split=5, n_jobs=-1))])
# best_score_ : 0.8459785558755861
# 소요 시간 : 11.42924427986145

model.fit(x_train, y_train)

print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)
print("소요 시간 :", time.time() - start_time)
