from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#         train_size=0.75, shuffle=True, random_state=66)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

kfold = KFold(5, shuffle=True, random_state=66)

# 2. 모델 구성

# model = LinearRegression()
# Acc :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]

# model = SVC()
# Acc :  [nan nan nan nan nan]

# model = KNeighborsClassifier()
# Acc :  [nan nan nan nan nan]

# model = KNeighborsRegressor()
# Acc :  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856]

# model = LogisticRegression()
# Acc :  [nan nan nan nan nan]

# model = DecisionTreeClassifier()
# Acc :  [nan nan nan nan nan]

# model = DecisionTreeRegressor()
# Acc :  [0.795421   0.67095659 0.78148487 0.73724642 0.74204975]

# model = RandomForestClassifier()
# Acc :  [nan nan nan nan nan]

# model = RandomForestRegressor()
# Acc :  [0.92173023 0.85706833 0.81713181 0.87809751 0.90898281]
# 평균 Acc : 1

scores = cross_val_score(model, x, y, cv=kfold)

from sklearn.metrics import r2_score, accuracy_score

print("Acc : ", scores)
# print("평균 Acc :", round(np.mean(scores))) # 평균값으로 수정
