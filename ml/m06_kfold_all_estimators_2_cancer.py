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

from sklearn.model_selection import train_test_split, KFold, cross_val_score

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                # train_size=0.8, shuffle=True, random_state=44)


# 2. 모델구성

allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')

# print(allAlgorithms)
# print(len(allAlgorithms)) # 41

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
AdaBoostClassifier [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 평균 : 0.9649
BaggingClassifier [0.92105263 0.94736842 0.94736842 0.9122807  0.96460177] 평균 : 0.9385
BernoulliNB [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 : 0.6274
CalibratedClassifierCV [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 평균 : 0.9263
CategoricalNB [nan nan nan nan nan] 평균 : nan
ClassifierChain 은 없는녀석
ComplementNB [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 평균 : 0.8963
DecisionTreeClassifier [0.93859649 0.92105263 0.92982456 0.92982456 0.96460177] 평균 : 0.9368
DummyClassifier [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 : 0.6274
ExtraTreeClassifier [0.90350877 0.93859649 0.89473684 0.92105263 0.91150442] 평균 : 0.9139
ExtraTreesClassifier [0.96491228 0.97368421 0.96491228 0.95614035 0.99115044] 평균 : 0.9702
GaussianNB [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 평균 : 0.942
GaussianProcessClassifier [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 평균 : 0.9122
GradientBoostingClassifier [0.94736842 0.96491228 0.95614035 0.93859649 0.97345133] 평균 : 0.9561
HistGradientBoostingClassifier [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 평균 : 0.9737
KNeighborsClassifier [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 평균 : 0.928
LabelPropagation [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 : 0.3902
LabelSpreading [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 : 0.3902
LinearDiscriminantAnalysis [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 평균 : 0.9614
LinearSVC [0.84210526 0.93859649 0.83333333 0.65789474 0.97345133] 평균 : 0.8491
LogisticRegression [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 평균 : 0.9385
LogisticRegressionCV [0.96491228 0.97368421 0.92105263 0.96491228 0.96460177] 평균 : 0.9578
MLPClassifier [0.90350877 0.93859649 0.92105263 0.94736842 0.9380531 ] 평균 : 0.9297
MultiOutputClassifier 은 없는녀석
MultinomialNB [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531] 평균 : 0.8928
NearestCentroid [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 평균 : 0.8893
NuSVC [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 평균 : 0.8735
OneVsOneClassifier 은 없는녀석
OneVsRestClassifier 은 없는녀석
OutputCodeClassifier 은 없는녀석
PassiveAggressiveClassifier [0.9122807  0.92982456 0.88596491 0.88596491 0.92035398] 평균 : 0.9069
Perceptron [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 평균 : 0.7771
QuadraticDiscriminantAnalysis [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 평균 : 0.9525
RadiusNeighborsClassifier [nan nan nan nan nan] 평균 : nan
RandomForestClassifier [0.97368421 0.95614035 0.96491228 0.96491228 0.97345133] 평균 : 0.9666
RidgeClassifier [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 평균 : 0.9543
RidgeClassifierCV [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 평균 : 0.9561
SGDClassifier [0.83333333 0.90350877 0.86842105 0.93859649 0.90265487] 평균 : 0.8893
SVC [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 평균 : 0.921
StackingClassifier 은 없는녀석
VotingClassifier 은 없는녀석
'''