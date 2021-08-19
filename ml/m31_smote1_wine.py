# pip install smote
from imblearn.over_sampling import SMOTE
from scipy.sparse import data
from sklearn import datasets
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings

warnings.filterwarnings('ignore')

datasets = load_wine()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)     # (178, 13) (178,)

# print(pd.Series(y).value_counts())      # value_counts() in pd
# 1    71
# 0    59
# 2    48

# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new =  x[:-39]
y_new =  y[:-39]

# print(x_new.shape, y_new.shape)     # (148, 13) (148,)

# print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new,
        train_size=0.75, shuffle=True, random_state=489, stratify=y_new
)

# print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score :", score)       # model.score : 0.9459459459459459

print("====================== smote 적용 ======================")

smote = SMOTE(random_state=66)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

model.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score :", score)       # model.score : 0.972972972972973

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())

# smote 전 : (111, 13) (111,)
# smote 후 : (159, 13) (159,)
# smote전 레이블 값 분포 :
#  1    53
# 0    44
# 2    14
# dtype: int64
# smote후 레이블 값 분포 :
#  0    53
# 1    53
# 2    53

# 증폭 이후 score가 증가한걸 알수있다.