import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# 1. 데이터

datasets = load_iris()

x = datasets.data
y = datasets.target

# train_test_split + k_fold

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                train_size=0.8, shuffle=True, random_state=94)

n_splits = 6
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# model = LinearSVC()
# Acc :  [1.   0.88 1.   1.   0.88 1.  ]
# 평균 Acc : 1

# Acc :  [1.   0.95 0.9  0.95 1.   1.  ]
# 평균 Acc : 1

# model = LogisticRegression()
# Acc :  [1.   0.96 0.96 1.   0.88 0.96]
# 평균 Acc : 1

# Acc :  [1.   0.95 1.   0.95 0.9  1.  ]
# 평균 Acc : 1

# model = SVC()
# Acc :  [1.   0.96 0.96 1.   0.92 1.  ]
# 평균 Acc : 1

# Acc :  [1.   0.95 1.   0.95 0.9  0.95]
# 평균 Acc : 1

# model = KNeighborsClassifier()
# Acc :  [0.96 0.96 1.   1.   0.88 0.96]
# 평균 Acc : 1

# Acc :  [1.   0.95 1.   0.95 0.95 0.95]
# 평균 Acc : 1

# model = DecisionTreeClassifier()
# Acc :  [0.96 0.96 1.   1.   0.88 0.92]
# 평균 Acc : 1

# Acc :  [0.9  0.9  1.   0.95 0.95 0.95]
# 평균 Acc : 1

# model = RandomForestClassifier()
# Acc :  [0.96 0.92 1.   1.   0.88 0.96]
# 평균 Acc : 1

# Acc :  [0.9  0.95 0.95 0.95 0.95 0.95]
# 평균 Acc : 1

# kfold 데이터를 cross_val_score, 이과정에 fit, score 포함

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("Acc : ", scores)
print("평균 Acc :", round(np.mean(scores))) # 평균값으로 수정