from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations

warnings.filterwarnings('ignore')

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.75, shuffle=True, random_state=66)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# 2. 모델 구성

# allAlgorithms = all_estimators(type_filter='classifier') # 전부 사용 불가능
allAlgorithms = all_estimators(type_filter='regressor')

# print(allAlgorithms)
print(len(allAlgorithms)) # 54

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, "의 정답률 :", acc)
    except:
        # continue
        print(name, "은 없는녀석")

'''
ARDRegression 은 없는녀석
AdaBoostRegressor 은 없는녀석
BaggingRegressor 은 없는녀석
BayesianRidge 은 없는녀석
CCA 은 없는녀석
DecisionTreeRegressor 은 없는녀석
DummyRegressor 은 없는녀석
ElasticNet 은 없는녀석
ElasticNetCV 은 없는녀석
ExtraTreeRegressor 은 없는녀석
ExtraTreesRegressor 은 없는녀석
GammaRegressor 은 없는녀석
GaussianProcessRegressor 은 없는녀석
GradientBoostingRegressor 은 없는녀석
HistGradientBoostingRegressor 은 없는녀석
HuberRegressor 은 없는녀석
IsotonicRegression 은 없는녀석
KNeighborsRegressor 은 없는녀석
KernelRidge 은 없는녀석
Lars 은 없는녀석
LarsCV 은 없는녀석
Lasso 은 없는녀석
LassoCV 은 없는녀석
LassoLars 은 없는녀석
LassoLarsCV 은 없는녀석
LassoLarsIC 은 없는녀석
LinearRegression 은 없는녀석
LinearSVR 은 없는녀석
MLPRegressor 은 없는녀석
MultiOutputRegressor 은 없는녀석
MultiTaskElasticNet 은 없는녀석
MultiTaskElasticNetCV 은 없는녀석
MultiTaskLasso 은 없는녀석
MultiTaskLassoCV 은 없는녀석
NuSVR 은 없는녀석
OrthogonalMatchingPursuit 은 없는녀석
OrthogonalMatchingPursuitCV 은 없는녀석
PLSCanonical 은 없는녀석
PLSRegression 은 없는녀석
PassiveAggressiveRegressor 은 없는녀석
PoissonRegressor 은 없는녀석
RANSACRegressor 은 없는녀석
RadiusNeighborsRegressor 은 없는녀석
RandomForestRegressor 은 없는녀석
RegressorChain 은 없는녀석
Ridge 은 없는녀석
RidgeCV 은 없는녀석
SGDRegressor 은 없는녀석
SVR 은 없는녀석
StackingRegressor 은 없는녀석
TheilSenRegressor 은 없는녀석
TransformedTargetRegressor 은 없는녀석
TweedieRegressor 은 없는녀석
VotingRegressor 은 없는녀석
'''