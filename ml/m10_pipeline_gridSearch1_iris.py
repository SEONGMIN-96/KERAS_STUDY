from math import pi
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
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

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline

n_splits = 6
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# 모델 : RandomForestRegressor

# parameter = [
#     {'randomforestclassifier__n_estimators' : [100, 200], 'randomforestclassifier__max_depth' : [6, 8, 10, 12]},
#     {'randomforestclassifier__max_depth' : [6, 8, 10, 12],'randomforestclassifier__min_samples_leaf' : [3, 5, 7 ,10]},
#     {'randomforestclassifier__min_samples_leaf' : [3, 5, 7 ,10],'randomforestclassifier__n_jobs' : [-1, 2, 4]},
#     {'randomforestclassifier__min_samples_split' : [2, 3, 5, 10],'randomforestclassifier__n_jobs' : [-1, 2, 4]},
#     {'randomforestclassifier__n_jobs' : [-1, 2, 4],'randomforestclassifier__n_estimators' : [100, 200]}
# ]

parameter = [
    {'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 8, 10, 12]},
    {'rf__max_depth' : [6, 8, 10, 12],'rf__min_samples_leaf' : [3, 5, 7 ,10]},
    {'rf__min_samples_leaf' : [3, 5, 7 ,10],'rf__n_jobs' : [-1, 2, 4]},
    {'rf__min_samples_split' : [2, 3, 5, 10],'rf__n_jobs' : [-1, 2, 4]},
    {'rf__n_jobs' : [-1, 2, 4],'rf__n_estimators' : [100, 200]}
]

start_time = time.time()

# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestClassifier())])

model = GridSearchCV(pipe, parameter, cv=kfold, verbose=1)

# model = RandomizedSearchCV(pipe, parameter, cv=kfold, verbose=1)

model.fit(x_train, y_train)

print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)
print("소요 시간 :", time.time() - start_time)

print("best_params_ :", model.best_params_)