from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from xgboost import plot_importance, XGBRegressor

# 1. 데이터

dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

# model = DecisionTreeRegressor()
# acc : -0.22034177430150925
# [0.0663183  0.00525457 0.2199695  0.12164122 0.04635836 0.0533545
#  0.05215741 0.00490397 0.35766135 0.07238081]

# model = RandomForestRegressor()
# acc : 0.37599731445986306
# [0.06542254 0.01197601 0.276521   0.10838415 0.04205815 0.05428557
#  0.04685184 0.01956506 0.29507077 0.07986492]

# model = GradientBoostingRegressor()

model = XGBRegressor()

# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가, 예측

acc = model.score(x_test, y_test)

# 5. 시각화

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

plot_importance(model)
plt.show()

print("acc :", acc)

print(model.feature_importances_)

