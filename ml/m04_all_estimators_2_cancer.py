from sklearn.utils import all_estimators
import warnings

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.datasets import load_breast_cancer

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

# 1. 데이터

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True, random_state=44)

# 1_1. 데이터 전처리

# from sklearn.preprocessing import QuantileTransformer

# scaler = QuantileTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# 2. 모델구성

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
AdaBoostClassifier 의 정답률 : 0.9736842105263158
BaggingClassifier 의 정답률 : 0.9649122807017544
BernoulliNB 의 정답률 : 0.6578947368421053
CalibratedClassifierCV 의 정답률 : 0.9824561403508771
CategoricalNB 은 없는녀석
ClassifierChain 은 없는녀석
ComplementNB 의 정답률 : 0.9473684210526315
DecisionTreeClassifier 의 정답률 : 0.9385964912280702
DummyClassifier 의 정답률 : 0.6578947368421053
ExtraTreeClassifier 의 정답률 : 0.9649122807017544
ExtraTreesClassifier 의 정답률 : 0.9736842105263158
GaussianNB 의 정답률 : 0.9736842105263158
GaussianProcessClassifier 의 정답률 : 0.9298245614035088
GradientBoostingClassifier 의 정답률 : 0.9736842105263158
HistGradientBoostingClassifier 의 정답률 : 0.9736842105263158
KNeighborsClassifier 의 정답률 : 0.956140350877193
LabelPropagation 의 정답률 : 0.3684210526315789
LabelSpreading 의 정답률 : 0.3684210526315789
LinearDiscriminantAnalysis 의 정답률 : 0.9912280701754386
LinearSVC 의 정답률 : 0.9385964912280702
LogisticRegression 의 정답률 : 0.9824561403508771
LogisticRegressionCV 의 정답률 : 0.9736842105263158
MLPClassifier 의 정답률 : 0.9122807017543859
MultiOutputClassifier 은 없는녀석
MultinomialNB 의 정답률 : 0.9473684210526315
NearestCentroid 의 정답률 : 0.9298245614035088
NuSVC 의 정답률 : 0.9385964912280702
OneVsOneClassifier 은 없는녀석
OneVsRestClassifier 은 없는녀석
OutputCodeClassifier 은 없는녀석
PassiveAggressiveClassifier 의 정답률 : 0.9649122807017544
Perceptron 의 정답률 : 0.8421052631578947
QuadraticDiscriminantAnalysis 의 정답률 : 0.9649122807017544
RadiusNeighborsClassifier 은 없는녀석
RandomForestClassifier 의 정답률 : 0.9649122807017544
RidgeClassifier 의 정답률 : 0.9824561403508771
RidgeClassifierCV 의 정답률 : 0.9824561403508771
SGDClassifier 의 정답률 : 0.9649122807017544
SVC 의 정답률 : 0.956140350877193
StackingClassifier 은 없는녀석
VotingClassifier 은 없는녀석
'''