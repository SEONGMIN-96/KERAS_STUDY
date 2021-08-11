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

# model = GridSearchCV(pipe, parameter, cv=kfold, verbose=1)
# Fitting 6 folds for each of 54 candidates, totalling 324 fits
# 최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('randomforestclassifier',
#                  RandomForestClassifier(max_depth=6))])
# best_score_ : 0.9791666666666666
# 소요 시간 : 64.59187006950378

model = RandomizedSearchCV(pipe, parameter, cv=kfold, verbose=1)
# Fitting 6 folds for each of 10 candidates, totalling 60 fits
# 최적의 매개변수 : Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('randomforestclassifier',
#                  RandomForestClassifier(max_depth=10, min_samples_leaf=5))])
# best_score_ : 0.9861111111111112
# 소요 시간 : 8.109821796417236

model.fit(x_train, y_train)

print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)
print("소요 시간 :", time.time() - start_time)
