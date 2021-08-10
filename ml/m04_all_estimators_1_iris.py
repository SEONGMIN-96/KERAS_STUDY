# from sklearn.utils.testing import all_estimators
import warnings
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')   

# 1. 데이터

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True, random_state=94
)

# 1_1. 데이터 전처리

from sklearn.preprocessing import QuantileTransformer

# scaler = QuantileTransformer()
# scaler.fit(x_train)
# scaler.transform(x_train)
# scaler.transform(x_test)

# 전처리 방식에따라 사용가능한 모델이 나뉨

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
AdaBoostClassifier 의 정답률 : 0.9666666666666667
BaggingClassifier 의 정답률 : 0.9666666666666667
BernoulliNB 의 정답률 : 0.26666666666666666
CalibratedClassifierCV 의 정답률 : 0.9333333333333333
CategoricalNB 의 정답률 : 0.9333333333333333
ClassifierChain 은 없는녀석
ComplementNB 의 정답률 : 0.6
DecisionTreeClassifier 의 정답률 : 0.9333333333333333
DummyClassifier 의 정답률 : 0.26666666666666666
ExtraTreeClassifier 의 정답률 : 0.9666666666666667
ExtraTreesClassifier 의 정답률 : 0.9666666666666667
GaussianNB 의 정답률 : 1.0
GaussianProcessClassifier 의 정답률 : 0.9666666666666667
GradientBoostingClassifier 의 정답률 : 0.9666666666666667
HistGradientBoostingClassifier 의 정답률 : 0.9333333333333333
KNeighborsClassifier 의 정답률 : 0.9666666666666667
LabelPropagation 의 정답률 : 0.9666666666666667
LabelSpreading 의 정답률 : 0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 : 0.9666666666666667
LinearSVC 의 정답률 : 0.9333333333333333
LogisticRegression 의 정답률 : 0.9666666666666667
LogisticRegressionCV 의 정답률 : 0.9666666666666667
MLPClassifier 의 정답률 : 0.9666666666666667
MultiOutputClassifier 은 없는녀석
MultinomialNB 의 정답률 : 0.8666666666666667
NearestCentroid 의 정답률 : 0.9666666666666667
NuSVC 의 정답률 : 0.9666666666666667
PassiveAggressiveClassifier 의 정답률 : 0.9
Perceptron 의 정답률 : 0.6333333333333333
QuadraticDiscriminantAnalysis 의 정답률 : 0.9666666666666667
RadiusNeighborsClassifier 의 정답률 : 1.0
RandomForestClassifier 의 정답률 : 0.9666666666666667
RidgeClassifier 의 정답률 : 0.9333333333333333
RidgeClassifierCV 의 정답률 : 0.9333333333333333
SGDClassifier 의 정답률 : 0.9333333333333333
SVC 의 정답률 : 0.9666666666666667
StackingClassifier 은 없는녀석
VotingClassifier 은 없는녀석
'''