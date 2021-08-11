from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#         train_size=0.75, shuffle=True, random_state=66)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=75)

# 2. 모델 구성

# model = LinearRegression()
# Acc :  [0.5760229  0.43913984 0.5307384  0.5113542  0.41231864]
# 평균 Acc : 0

# model = SVC()
# Acc :  [0.02247191 0.         0.         0.         0.01136364]
# 평균 Acc : 0

# model = KNeighborsClassifier()
# Acc :  [0. 0. 0. 0. 0.]
# 평균 Acc : 0

# model = KNeighborsRegressor()
# Acc :  [0.40191064 0.24976359 0.44783313 0.45674965 0.28987079]
# 평균 Acc : 0

# model = LogisticRegression()
# Acc :  [0.01123596 0.         0.01136364 0.         0.01136364]
# 평균 Acc : 0

# model = DecisionTreeClassifier()
# Acc :  [0. 0. 0. 0. 0.]
# 평균 Acc : 0

# model = DecisionTreeRegressor()
# Acc :  [-0.14403678 -0.34001504 -0.02665136 -0.08866939 -0.52946604]
# 평균 Acc : 0

# model = RandomForestClassifier()
# Acc :  [0.         0.         0.01136364 0.         0.02272727]
# 평균 Acc : 0

# model = RandomForestRegressor()
# Acc :  [0.48043363 0.36581169 0.48410972 0.50516211 0.33636042]
# 평균 Acc : 0

scores = cross_val_score(model, x, y, cv=kfold)

print("Acc : ", scores)
print("평균 Acc :", round(np.mean(scores))) # 평균값으로 수정