from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# 2. 모델 구성

# model = LinearRegression()
# ValueError: continuous is not supported
# model.score :  0.8171699812170787

# model = SVC()
# ValueError: Unknown label type: 'continuous'

# model = KNeighborsClassifier()
# ValueError: Unknown label type: 'continuous'

# model = KNeighborsRegressor()
# ValueError: continuous is not supported
# model.score :  0.5972314424994741

# model = LogisticRegression()
# ValueError: Unknown label type: 'continuous'

# model = DecisionTreeClassifier()
# ValueError: Unknown label type: 'continuous'

# model = DecisionTreeRegressor()
# model.score :  0.7305869673880918

# model = RandomForestClassifier()
# ValueError: Unknown label type: 'continuous'

# model = RandomForestRegressor()
# model.score :  0.8977946642292762

# 3. 컴파일, 훈련

model.fit(x_train, y_train)

# 4. 평가, 예측

from sklearn.metrics import r2_score, accuracy_score

results = model.score(x_test, y_test)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)

print("model.score : ",results)
# print("accuracy_score : ",acc)
