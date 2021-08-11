# from sklearn.utils.testing import all_estimators
import warnings
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
import warnings
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings('ignore')   

# 1. 데이터

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# 전처리 방식에따라 사용가능한 모델이 나뉨

# 2. 모델 구성

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
AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667] 평균 : 0.8867
BaggingClassifier [0.9        0.96666667 1.         0.9        0.96666667] 평균 : 0.9467
BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 : 0.2933
CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667] 평균 : 0.9133
CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ] 평균 : 0.9333
ClassifierChain 은 없는녀석
ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ] 평균 : 0.6667
DecisionTreeClassifier [0.96666667 0.96666667 1.         0.9        0.93333333] 평균 : 0.9533
DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 : 0.2933
ExtraTreeClassifier [0.96666667 1.         0.93333333 0.9        0.93333333] 평균 : 0.9467
ExtraTreesClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667] 평균 : 0.9467
GaussianNB [0.96666667 0.9        1.         0.9        0.96666667] 평균 : 0.9467
GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 : 0.96
GradientBoostingClassifier [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 : 0.9667
HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667] 평균 : 0.94
KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 : 0.96
LabelPropagation [0.93333333 1.         1.         0.9        0.96666667] 평균 : 0.96
LabelSpreading [0.93333333 1.         1.         0.9        0.96666667] 평균 : 0.96
LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ] 평균 : 0.98
LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ] 평균 : 0.9667
LogisticRegression [1.         0.96666667 1.         0.9        0.96666667] 평균 : 0.9667
LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ] 평균 : 0.9733
MLPClassifier [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 : 0.9733
MultiOutputClassifier 은 없는녀석
MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ] 평균 : 0.9667
NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667] 평균 : 0.9333
NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 : 0.9733
OneVsOneClassifier 은 없는녀석
OneVsRestClassifier 은 없는녀석
OutputCodeClassifier 은 없는녀석
PassiveAggressiveClassifier [0.73333333 0.86666667 0.86666667 0.66666667 0.96666667] 평균 : 0.82
Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 평균 : 0.78
QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.        ] 평균 : 0.98
RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.        ] 평균 : 0.9533
RandomForestClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667] 평균 : 0.9467
RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 : 0.84
RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 : 0.84
SGDClassifier [0.66666667 0.96666667 0.86666667 0.66666667 0.9       ] 평균 : 0.8133
SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 : 0.9667
StackingClassifier 은 없는녀석
VotingClassifier 은 없는녀석
'''