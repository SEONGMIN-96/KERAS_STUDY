from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 데이터

dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

# model = DecisionTreeClassifier(max_depth=4)
# acc : 0.9666666666666667
# [0.         0.0125026  0.53835801 0.44913938]

# model = RandomForestClassifier()
# acc : 0.9333333333333333
# [0.12362616 0.03344172 0.44531126 0.39762086]

# model = GradientBoostingClassifier()
# acc : 0.9333333333333333
# [0.00298427 0.01506014 0.34245013 0.63950545]

# model = XGBClassifier()
# acc : 0.9
# [0.01835513 0.0256969  0.62045246 0.3354955 ]

# 3. 훈련

# model.fit(x_train, y_train)

# # 4. 평가, 예측

# acc = model.score(x_test, y_test)

# # 5. 시각화

# def plot_feature_importances_dataset(model):
#     n_features = dataset.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#                 align='center')
#     plt.yticks(np.arange(n_features), dataset.feature_names)
#     plt.xlabel("Feature Importancs")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

# print("acc :", acc)

# print(model.feature_importances_)



###################################################################

# 중요도 하위 20%의 컬럼 제거 후, 모델 비교

dataset_data = pd.DataFrame(dataset.data)

dataset_data = dataset_data.drop([0], axis=1)
dataset_data = np.array(dataset_data)

x_train, x_test, y_train, y_test = train_test_split(dataset_data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

# model = DecisionTreeClassifier(max_depth=4)
# 기존
# acc : 0.9666666666666667
# [0.         0.0125026  0.53835801 0.44913938]

# 제거 후
# acc : 0.9666666666666667
# [0.0125026  0.03213177 0.95536562]

# model = RandomForestClassifier()
# 기존
# acc : 0.9333333333333333
# [0.12362616 0.03344172 0.44531126 0.39762086]

# 제거
# acc : 0.9
# [0.14025948 0.40934996 0.45039056]

# model = GradientBoostingClassifier()
# 기존
# acc : 0.9333333333333333
# [0.00298427 0.01506014 0.34245013 0.63950545]

# 제거
# acc : 0.9333333333333333
# [0.01333957 0.3109683  0.67569213]

model = XGBClassifier()
# 기존
# acc : 0.9
# [0.01835513 0.0256969  0.62045246 0.3354955 ]

# 제거
# acc : 0.9
# [0.02876593 0.63379896 0.3374351 ]

# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가, 예측

acc = model.score(x_test, y_test)

print("acc :", acc)
print(model.feature_importances_)

# 모델별로 importance값이 다른경우가 존재하기 때문에 (많이는아님) 어떠한 모델을 사용하는지에따라
# 소거해줄 coulmns을 결정해줘야한다.

# importance값이 애초에 0일경우 소거해주어도 결과값에 변화를 주진못한다.
# 하지만 속도에는 영향을 미칠것이다.