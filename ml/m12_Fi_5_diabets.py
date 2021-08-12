from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# 1. 데이터

dataset = load_diabetes()
# x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
#                     train_size=0.8, random_state=66
# )

# # 2. 모델

# model = DecisionTreeRegressor()
# # 기존
# # acc : -0.22034177430150925
# # [0.0663183  0.00525457 0.2199695  0.12164122 0.04635836 0.0533545
# #  0.05215741 0.00490397 0.35766135 0.07238081]

# model = RandomForestRegressor()
# # 기존
# # acc : 0.37599731445986306
# # [0.06542254 0.01197601 0.276521   0.10838415 0.04205815 0.05428557
# #  0.04685184 0.01956506 0.29507077 0.07986492]

# model = GradientBoostingRegressor()
# # 기존
# # acc : 0.3893758428656644
# # [0.05955666 0.01142002 0.2765704  0.11853846 0.02502413 0.05353008
# #  0.04007051 0.0171977  0.34186898 0.05622307]

# model = XGBRegressor()
# # acc : 0.23802704693460175
# # [0.02593722 0.03821947 0.19681752 0.06321313 0.04788675 0.05547737
# #  0.07382318 0.03284872 0.3997987  0.06597802]

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

dataset_data = dataset_data.drop([0,1], axis=1)
dataset_data = np.array(dataset_data)

x_train, x_test, y_train, y_test = train_test_split(dataset_data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

model = DecisionTreeRegressor()
# 기존
# acc : -0.22034177430150925
# [0.0663183  0.00525457 0.2199695  0.12164122 0.04635836 0.0533545
#  0.05215741 0.00490397 0.35766135 0.07238081]

# 제거
# acc : -0.4546749012451661
# [0.22362281 0.11985426 0.02231221 0.08919216 0.07287213 0.00844756
#  0.39307287 0.070626  ]

# model = RandomForestRegressor()
# 기존
# acc : 0.37599731445986306
# [0.06542254 0.01197601 0.276521   0.10838415 0.04205815 0.05428557
#  0.04685184 0.01956506 0.29507077 0.07986492]

# 제거
# acc : 0.37707813372183596
# [0.26867289 0.11906509 0.0501814  0.06837294 0.06018906 0.01954514
#  0.33054265 0.08343084]

# model = GradientBoostingRegressor()
# 기존
# acc : 0.3893758428656644
# [0.05955666 0.01142002 0.2765704  0.11853846 0.02502413 0.05353008
#  0.04007051 0.0171977  0.34186898 0.05622307]

# 제거
# acc : 0.3622541621540811
# [0.29235266 0.11933569 0.04092139 0.07324818 0.03821501 0.0205965
#  0.34820958 0.06712097]

# model = XGBRegressor()
# 기존
# acc : 0.23802704693460175
# [0.02593722 0.03821947 0.19681752 0.06321313 0.04788675 0.05547737
#  0.07382318 0.03284872 0.3997987  0.06597802]

# 제거
# acc : 0.24248670495525249
# [0.15730305 0.06859756 0.05361945 0.0800518  0.05962749 0.18795057
#  0.31990343 0.07294661]

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