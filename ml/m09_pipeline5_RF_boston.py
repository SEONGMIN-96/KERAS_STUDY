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

from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline, Pipeline

warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

model = make_pipeline(MinMaxScaler(), SVC())

start_time = time.time()

model.fit(x_train, y_train)

print("model.score :", model.score(x_test, y_test))
print("소요 시간 :", time.time() - start_time)
