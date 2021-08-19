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

datasets = pd.read_csv('./_data/wine_quality/winequality-white.csv',
                        index_col=None, header=0, sep=';')

datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]
print(x.shape, y.shape)     # 

print(pd.Series(y).value_counts())      # value_counts() in pd

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.75, shuffle=True, random_state=489, stratify=y
)

print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score :", score)       # 

print("====================== smote 적용 ======================")

smote = SMOTE(random_state=66, k_neighbors=3)

x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train)

model.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score :", score)       # 

print("smote 전 :", x_train.shape, y_train.shape)
print("smote 후 :", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())

# k_neighbors 조정으로 낮은 라벨값에도 적용이 가능하다.
# 하지만 연결값이 낮기때문에 좋은 결과를 기대하긴 힘들다.