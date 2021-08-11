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

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

model = make_pipeline(MinMaxScaler(), SVC())

start_time = time.time()

model.fit(x_train, y_train)

print("model.score :", model.score(x_test, y_test))
print("소요 시간 :", time.time() - start_time)