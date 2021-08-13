from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터

datasets = load_boston()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                    train_size=0.8, random_state=66
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델

model = XGBRegressor(n_estimators=10000, learning_rate=0.01,
                     n_jobs=5, tree_method='gpu_hist',
                     gpu_id=0,
                     predictor='gpu_predictor'     # cpu_predictor
)

# 3. 훈련

import time

start_time = time.time()
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'],
          eval_set=[(x_train, y_train),(x_test, y_test)],
)

print("소요시간 :", time.time()-start_time)

'''
i7-9700, 2080
njobs=1 : 소요시간 : 7.574501991271973
njobs=2 : 소요시간 : 7.657411336898804
njobs=3 : 소요시간 : 7.695486783981323
njobs=8 : 소요시간 : 7.698853969573975
'''