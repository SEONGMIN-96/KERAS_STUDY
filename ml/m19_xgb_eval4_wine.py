from sklearn import datasets
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터

datasets = load_wine()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)     # (178, 13) (178,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                    train_size=0.8, random_state=66
)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델

model = XGBClassifier(n_estimators=5000, learning_rate=0.0005, n_jobs=1)

# 3. 훈련

model.fit(x_train, y_train, verbose=1, eval_metric=['merror', 'mlogloss'],
          eval_set=[(x_train, y_train),(x_test, y_test)]
)

# 4. 평가

results = model.score(x_test, y_test)
print("results :", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 :", r2)

'''
results : 1.0
r2 : 1.0
'''

print("===========================")
hist = model.evals_result()
print(hist['validation_0']['merror'])
print(hist['validation_1']['merror'])

import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))
plt.subplot(211)
plt.ylabel('merror')
plt.plot(hist['validation_0']['merror'], '-', label='val_0')
plt.plot(hist['validation_1']['merror'], '-', label='val_1')
plt.legend(loc=0)

plt.subplot(212)
plt.ylabel('mlogloss')
plt.plot(hist['validation_0']['mlogloss'], '-', label='val_0')
plt.plot(hist['validation_1']['mlogloss'], '-', label='val_1')
plt.legend(loc=0)

plt.show()