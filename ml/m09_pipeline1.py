import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
import time

# 1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True, random_state=94
)

# 1_1. 데이터 전처리

from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler, MaxAbsScaler

# 2. 모델 구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline, Pipeline

model = make_pipeline(MinMaxScaler(), SVC())

start_time = time.time()

model.fit(x_train, y_train)

print("model.score :", model.score(x_test, y_test))
print("소요 시간 :", time.time() - start_time)
