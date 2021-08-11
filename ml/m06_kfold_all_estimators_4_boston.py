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
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.python.keras import activations

warnings.filterwarnings('ignore')

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
# scaler = PowerTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# 2. 모델 구성

# allAlgorithms = all_estimators(type_filter='classifier') # 전부 사용 불가능
allAlgorithms = all_estimators(type_filter='regressor')

# print(allAlgorithms)
print(len(allAlgorithms)) # 54

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, "평균 :", round(np.mean(scores), 4))
    except:
        # continue
        print(name, "은 없는녀석")

'''
ARDRegression [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 평균 : 0.6985
AdaBoostRegressor [0.89388886 0.8075849  0.80299333 0.85061842 0.85769015] 평균 : 0.8426
BaggingRegressor [0.92517145 0.85653185 0.80758251 0.89028678 0.8896367 ] 평균 : 0.8738
BayesianRidge [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 평균 : 0.7038
CCA [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 평균 : 0.6471
DecisionTreeRegressor [0.81121487 0.75008986 0.79805561 0.7338372  0.81367139] 평균 : 0.7814
DummyRegressor [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 : -0.0135
ElasticNet [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 평균 : 0.6708
ElasticNetCV [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 평균 : 0.6565
ExtraTreeRegressor [0.68267635 0.74775526 0.56725853 0.83703043 0.63904954] 평균 : 0.6948
ExtraTreesRegressor [0.93381762 0.85194964 0.78720716 0.87645226 0.92523339] 평균 : 0.8749
GammaRegressor [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] 평균 : -0.0136
GaussianProcessRegressor [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] 평균 : -5.9286
GradientBoostingRegressor [0.94581241 0.8326186  0.82655551 0.88553376 0.93058744] 평균 : 0.8842
HistGradientBoostingRegressor [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 평균 : 0.8581
HuberRegressor [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398] 평균 : 0.584
IsotonicRegression [nan nan nan nan nan] 평균 : nan
KNeighborsRegressor [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 : 0.5286
KernelRidge [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 평균 : 0.6854
Lars [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 평균 : 0.6977
LarsCV [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 평균 : 0.6928
Lasso [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 평균 : 0.6657
LassoCV [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 평균 : 0.6779
LassoLars [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 : -0.0135
LassoLarsCV [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 평균 : 0.6965
LassoLarsIC [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 평균 : 0.713
LinearRegression [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 : 0.7128
LinearSVR [0.47079623 0.68534102 0.21097303 0.53696526 0.34913098] 평균 : 0.4506
MLPRegressor [0.58858452 0.64350241 0.44217229 0.36676265 0.49585394] 평균 : 0.5074
MultiOutputRegressor 은 없는녀석
MultiTaskElasticNet [nan nan nan nan nan] 평균 : nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 : nan
MultiTaskLasso [nan nan nan nan nan] 평균 : nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 : nan
NuSVR [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 평균 : 0.2295
OrthogonalMatchingPursuit [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 평균 : 0.5343
OrthogonalMatchingPursuitCV [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 평균 : 0.6578
PLSCanonical [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] 평균 : -2.2096
PLSRegression [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 평균 : 0.6847
PassiveAggressiveRegressor [-0.05963945  0.0823183  -0.55979331 -3.01731899 -0.30429821] 평균 : -0.7717
PoissonRegressor [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656] 평균 : 0.7549
RANSACRegressor [0.41146804 0.54811723 0.42594791 0.00263654 0.52084972] 평균 : 0.3818
RadiusNeighborsRegressor [nan nan nan nan nan] 평균 : nan
RandomForestRegressor [0.92133178 0.85454339 0.82861712 0.87810223 0.90427628] 평균 : 0.8774
RegressorChain 은 없는녀석
Ridge [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 평균 : 0.7109
RidgeCV [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912] 평균 : 0.7128
SGDRegressor [-2.10942233e+26 -9.14397919e+26 -5.49560875e+25 -4.72151595e+25
 -2.34702001e+26] 평균 : -2.924426801041616e+26
SVR [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 평균 : 0.1963
StackingRegressor 은 없는녀석
TheilSenRegressor [0.79032566 0.73020725 0.59272068 0.54984677 0.71423095] 평균 : 0.6755
TransformedTargetRegressor [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 : 0.7128
TweedieRegressor [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475] 평균 : 0.6558
VotingRegressor 은 없는녀석
'''