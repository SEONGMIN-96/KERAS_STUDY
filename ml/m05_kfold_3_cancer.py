from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

from operator import mod
from re import T
import numpy as np
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

# 1. 데이터

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#                                 train_size=0.8, shuffle=True, random_state=44)

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

# 1_1. 데이터 전처리

# from sklearn.preprocessing import QuantileTransformer

# scaler = QuantileTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# 2. 모델 구성

# model = LinearSVC()
# Acc :  [0.90350877 0.93859649 0.85087719 0.79824561 0.94690265]
# 평균 Acc : 1

# model = SVC()
# Acc :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
# 평균 Acc : 1

# model = KNeighborsClassifier()
# Acc :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
# 평균 Acc : 1

# model = LogisticRegression()
# Acc :  [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177]
# 평균 Acc : 1

# model = DecisionTreeClassifier()
# Acc :  [0.90350877 0.92982456 0.92105263 0.87719298 0.95575221]
# 평균 Acc : 1

# model = RandomForestClassifier()
# Acc :  [0.96491228 0.95614035 0.96491228 0.94736842 0.97345133]
# 평균 Acc : 1

scores = cross_val_score(model, x, y, cv=kfold)

print("Acc : ", scores)
print("평균 Acc :", round(np.mean(scores))) # 평균값으로 수정
