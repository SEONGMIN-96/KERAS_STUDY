from sklearn.utils import all_estimators
import warnings
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

# 1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                train_size=0.8, shuffle=True)

# 1_2. preprocessing

# from sklearn.preprocessing import QuantileTransformer

# scaler = QuantileTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# 2. 모델 구성

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

# print(allAlgorithms)
# print(len(allAlgorithms)) # 41

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
AdaBoostClassifier 의 정답률 : 0.9722222222222222
BaggingClassifier 의 정답률 : 0.9444444444444444
BernoulliNB 의 정답률 : 0.4722222222222222
CalibratedClassifierCV 의 정답률 : 1.0
CategoricalNB 은 없는녀석
ClassifierChain 은 없는녀석
ComplementNB 의 정답률 : 0.7222222222222222
DecisionTreeClassifier 의 정답률 : 0.8611111111111112
DummyClassifier 의 정답률 : 0.4722222222222222
ExtraTreeClassifier 의 정답률 : 0.7222222222222222
ExtraTreesClassifier 의 정답률 : 0.9722222222222222
GaussianNB 의 정답률 : 1.0
GaussianProcessClassifier 의 정답률 : 0.5277777777777778
GradientBoostingClassifier 의 정답률 : 0.9722222222222222
HistGradientBoostingClassifier 의 정답률 : 0.9722222222222222
KNeighborsClassifier 의 정답률 : 0.6388888888888888
LabelPropagation 의 정답률 : 0.5
LabelSpreading 의 정답률 : 0.5
LinearDiscriminantAnalysis 의 정답률 : 1.0
LinearSVC 의 정답률 : 0.75
LogisticRegression 의 정답률 : 1.0
LogisticRegressionCV 의 정답률 : 0.9444444444444444
MLPClassifier 의 정답률 : 0.9166666666666666
MultiOutputClassifier 은 없는녀석
MultinomialNB 의 정답률 : 0.8611111111111112
NearestCentroid 의 정답률 : 0.75
NuSVC 의 정답률 : 0.9166666666666666
OneVsOneClassifier 은 없는녀석
OneVsRestClassifier 은 없는녀석
OutputCodeClassifier 은 없는녀석
PassiveAggressiveClassifier 의 정답률 : 0.6111111111111112
Perceptron 의 정답률 : 0.3333333333333333
QuadraticDiscriminantAnalysis 의 정답률 : 1.0
RadiusNeighborsClassifier 은 없는녀석
RandomForestClassifier 의 정답률 : 0.9444444444444444
RidgeClassifier 의 정답률 : 1.0
RidgeClassifierCV 의 정답률 : 1.0
SGDClassifier 의 정답률 : 0.6666666666666666
SVC 의 정답률 : 0.75
StackingClassifier 은 없는녀석
VotingClassifier 은 없는녀석
'''