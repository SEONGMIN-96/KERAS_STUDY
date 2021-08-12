from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# 1. 데이터

dataset = load_boston()
# x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
#                     train_size=0.8, random_state=66
# )

# # 2. 모델

# model = DecisionTreeRegressor(max_depth=4)
# # 기존
# # acc : 0.8774175457631728
# # [0.03878833 0.         0.         0.         0.00802925 0.29639913
# #  0.         0.05954596 0.         0.01862509 0.         0.
# #  0.57861225]

# model = RandomForestRegressor()
# # 기존
# # acc : 0.9226730640546915
# # [0.03777992 0.00114303 0.00699198 0.00130014 0.02562012 0.42510804
# #  0.01463562 0.06779926 0.00459551 0.01272414 0.01591433 0.01178495
# #  0.37460295]

# model = GradientBoostingRegressor()
# # 기존
# # acc : 0.945399460920079
# # [2.43294931e-02 2.48094565e-04 2.01448294e-03 1.80771482e-04
# #  4.24357723e-02 3.58355358e-01 6.39212463e-03 8.25636756e-02
# #  2.40815286e-03 1.11715891e-02 3.38537499e-02 6.57992587e-03
# #  4.29466809e-01]

# model = XGBRegressor()
# # 기존
# # acc : 0.9221188601856797
# # [0.01447935 0.00363372 0.01479119 0.00134153 0.06949984 0.30128643
# #  0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
# #  0.42848358]

# # 3. 훈련

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

dataset_data = dataset_data.drop([1,2,3], axis=1)
dataset_data = np.array(dataset_data)

x_train, x_test, y_train, y_test = train_test_split(dataset_data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

model = DecisionTreeRegressor(max_depth=4)
# 기존
# acc : 0.8774175457631728
# [0.03878833 0.         0.         0.         0.00802925 0.29639913
#  0.         0.05954596 0.         0.01862509 0.         0.
#  0.57861225]

# 제거
# acc : 0.8774175457631728
# [0.03878833 0.0262834  0.29639913 0.         0.05991689 0.
#  0.         0.         0.         0.57861225]

# model = RandomForestRegressor()
# 기존
# acc : 0.9226730640546915
# [0.03777992 0.00114303 0.00699198 0.00130014 0.02562012 0.42510804
#  0.01463562 0.06779926 0.00459551 0.01272414 0.01591433 0.01178495
#  0.37460295]

# 제거
# acc : 0.925551842106425
# [0.04177515 0.02008381 0.3730908  0.01587839 0.06544856 0.00453639
#  0.01639224 0.01753946 0.0126735  0.43258169]

# model = GradientBoostingRegressor()
# 기존
# acc : 0.945399460920079
# [2.43294931e-02 2.48094565e-04 2.01448294e-03 1.80771482e-04
#  4.24357723e-02 3.58355358e-01 6.39212463e-03 8.25636756e-02
#  2.40815286e-03 1.11715891e-02 3.38537499e-02 6.57992587e-03
#  4.29466809e-01]

# 제거
# acc : 0.9442978394607116
# [0.02733843 0.04275471 0.35536878 0.0060161  0.08353293 0.00306628
#  0.01254165 0.03362565 0.00670391 0.42905156]

# model = XGBRegressor()
# 기존
# acc : 0.9221188601856797
# [0.01447935 0.00363372 0.01479119 0.00134153 0.06949984 0.30128643
#  0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
#  0.42848358]

# 제거
# acc : 0.9230605699784487
# [0.01807656 0.05122639 0.29952207 0.01511599 0.06779128 0.0147081
#  0.0404624  0.0373815  0.01073553 0.4449802 ]

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