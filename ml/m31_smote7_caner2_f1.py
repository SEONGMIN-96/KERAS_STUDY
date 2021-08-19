# 지표는 f1
# 라벨 0 을 112개 삭제

from imblearn.over_sampling import SMOTE
from scipy.sparse import data
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

# print(x.shape, y.shape)

# print(pd.Series(y).value_counts())      # value_counts() in pd

'''
# print(x.shape, y.shape)     #

# print(pd.Series(y).value_counts())      # value_counts() in pd

############################################################
###### 라벨 대통합 !!
###########################################################

print("===============================================")

for i in list(range(len(y))):
        if y[i] == 9:
                y[i] = 2
        elif y[i] == 8:
                y[i] = 2
        elif y[i] == 7:
                y[i] = 2
        elif y[i] == 6:
                y[i] = 1
        elif y[i] == 5:
                y[i] = 0
        elif y[i] == 4:
                y[i] = 0
        elif y[i] == 3:
                y[i] = 0

# for index, value in enumerate(y):
#         if value == 9:
#                 y[index] = 8

print(pd.Series(y).value_counts())
'''

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66#, stratify=y
)

print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score :", score) 

y_pred = model.predict(x_test) 
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score :", f1)

print("====================== smote 적용 ======================")

smote = SMOTE(random_state=66, k_neighbors=15)

start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)
end_time = time.time() - start_time

model.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score :", score)       # 

y_pred = model.predict(x_test) 
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score :", f1)

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())

print("소요시간 :", end_time)
