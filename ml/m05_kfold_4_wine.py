from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

import numpy as np
from sklearn.datasets import load_wine

warnings.filterwarnings('ignore')

# 1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

# x_train, x_test, y_train, y_test = train_test_split(x, y,
                # train_size=0.8, shuffle=True)

kfold = KFold(5, shuffle=True, random_state=66)

# 2. 모델 구성

# model = LinearSVC()
# Acc :  [0.86111111 0.77777778 0.91666667 0.85714286 0.88571429]
# 평균 Acc : 1

# model = SVC()
# Acc :  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ]
# 평균 Acc : 1

# model = KNeighborsClassifier()
# Acc :  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714]
# 평균 Acc : 1

# model = LogisticRegression()
# Acc :  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ]
# 평균 Acc : 1

# model = DecisionTreeClassifier()
# Acc :  [0.97222222 0.97222222 0.91666667 0.88571429 0.91428571]
# 평균 Acc : 1

# model = RandomForestClassifier()
# Acc :  [1.         0.94444444 1.         1.         1.        ]
# 평균 Acc : 1

scores = cross_val_score(model, x, y, cv=kfold)

print("Acc : ", scores)
print("평균 Acc :", round(np.mean(scores))) # 평균값으로 수정

