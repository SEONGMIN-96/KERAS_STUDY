# 실습
# 아웃라이어 확인!!

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

datasets = pd.read_csv('./_data/wine_quality/winequality-white.csv',
                        index_col=None, header=0, sep=';')

# print(datasets.head())
# print(datasets.shape)       # (4898, 12)
# print(datasets.describe())
# print(datasets.info())

# q3 = np.where(datasets['quality'] == 3)
# q4 = np.where(datasets['quality'] == 4)
# q5 = np.where(datasets['quality'] == 5)
# q6 = np.where(datasets['quality'] == 6)
# q7 = np.where(datasets['quality'] == 7)
# q8 = np.where(datasets['quality'] == 8)
# q9 = np.where(datasets['quality'] == 9)

# quality = ['q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9']
# count = [q3[0].shape[0], q4[0].shape[0], q5[0].shape[0], q6[0].shape[0],
#          q7[0].shape[0], q8[0].shape[0], q9[0].shape[0]     
# ]

count_data = datasets.groupby('quality')['quality'].count()
print(count_data)

'''
datasets = datasets.values      # numpy로 바꿀때 values쓰면 편함
# print(type(datasets))
# print(datasets.shape)         # (4898, 12)

x = datasets[:, :11]
y = datasets[:, 11]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size=0.8, shuffle=True, random_state=66
)

'''

import matplotlib.pyplot as plt

# datasets의 바 그래프를 그리시오!!

# plt.bar(quality, count)
# plt.show()

plt.bar(count_data.index, count_data)
plt.show()

'''
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print("model.score :", score)       # model.score : 0.6816326530612244
'''
