import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target

pca = PCA(n_components=7)
x = pca.fit_transform(x)

print(x.shape, y.shape)         # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,   
        train_size=0.8, shuffle=True, random_state=66
)

# 2. 모델

from xgboost import XGBRegressor

model = XGBRegressor()

# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가, 예측

results = model.score(x_test, y_test)
print("결과 :", results)

# 기존 결과 : 0.23802704693460175 
# PCA 결과 : 0.3210924574289413