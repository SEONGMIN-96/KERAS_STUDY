from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터

dataset = load_wine()
# x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
#                     train_size=0.8, random_state=66
# )

# # 2. 모델

# model = DecisionTreeClassifier()
# # 기존
# # acc : 0.9444444444444444
# # [0.         0.00489447 0.         0.0555874  0.05677108 0.
# #  0.1569445  0.         0.         0.         0.03045446 0.33215293
# #  0.36319516]

# model = RandomForestClassifier()
# # 기존
# # acc : 1.0
# # [0.10503536 0.02598386 0.01388783 0.02662012 0.02931552 0.04756739
# #  0.17229227 0.01295594 0.02403751 0.17308295 0.09723877 0.0941247
# #  0.17785779]

# model = GradientBoostingClassifier()
# # 기존
# # acc : 0.9722222222222222
# # [1.95566638e-02 3.83339043e-02 2.08543821e-02 9.43332746e-03
# #  2.07404494e-03 2.81157204e-05 1.06920546e-01 1.26194365e-04
# #  7.27881908e-05 2.50819063e-01 2.35869818e-02 2.52969281e-01
# #  2.75224707e-01]

# model = XGBClassifier()
# # 기존
# # acc : 1.0
# # [0.01854127 0.04139536 0.01352911 0.01686821 0.02422602 0.00758254
# #  0.10707161 0.01631111 0.00051476 0.12775211 0.01918284 0.50344414
# #  0.10358089]

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

dataset_data = dataset_data.drop([0,2,5], axis=1)
dataset_data = np.array(dataset_data)

x_train, x_test, y_train, y_test = train_test_split(dataset_data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

model = DecisionTreeClassifier()
# 기존
# acc : 0.9444444444444444
# [0.         0.00489447 0.         0.0555874  0.05677108 0.
#  0.1569445  0.         0.         0.         0.03045446 0.33215293
#  0.36319516]

# 제거
# acc : 0.9166666666666666
# [0.00489447 0.0555874  0.04078249 0.1569445  0.         0.
#  0.04644305 0.         0.33215293 0.36319516]

# model = RandomForestClassifier()
# 기존
# acc : 1.0
# [0.10503536 0.02598386 0.01388783 0.02662012 0.02931552 0.04756739
#  0.17229227 0.01295594 0.02403751 0.17308295 0.09723877 0.0941247
#  0.17785779]

# 제거
# acc : 1.0
# [0.04107557 0.05272895 0.03138227 0.19429615 0.00899358 0.01860038
#  0.20082528 0.11458379 0.12804087 0.20947316]

# model = GradientBoostingClassifier()
# 기존
# acc : 0.9722222222222222
# [1.95566638e-02 3.83339043e-02 2.08543821e-02 9.43332746e-03
#  2.07404494e-03 2.81157204e-05 1.06920546e-01 1.26194365e-04
#  7.27881908e-05 2.50819063e-01 2.35869818e-02 2.52969281e-01
#  2.75224707e-01]

# 제거
# acc : 0.9722222222222222
# [5.20990096e-02 6.75320403e-03 3.59318804e-03 1.17904150e-01
#  1.06503155e-04 7.43450415e-05 2.67749923e-01 2.88031468e-02
#  2.37649715e-01 2.85266815e-01]

# model = XGBClassifier()
# 기존
# acc : 1.0
# [0.01854127 0.04139536 0.01352911 0.01686821 0.02422602 0.00758254
#  0.10707161 0.01631111 0.00051476 0.12775211 0.01918284 0.50344414
#  0.10358089]

# 제거
# acc : 1.0
# [0.04444635 0.02415159 0.02414252 0.15546806 0.02677766 0.00089528
#  0.15274441 0.02376685 0.4220517  0.12555557]

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