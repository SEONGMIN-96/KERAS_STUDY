from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터

dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

# model = DecisionTreeClassifier()
# acc : 0.9298245614035088
# [0.         0.05940707 0.         0.         0.         0.
#  0.         0.01967507 0.         0.00468454 0.01233852 0.
#  0.         0.         0.01405362 0.02248579 0.         0.00433754
#  0.         0.         0.         0.01612033 0.         0.71474329
#  0.00468454 0.         0.00461856 0.11660508 0.         0.00624605]

# model = RandomForestClassifier()
# acc : 0.956140350877193
# [0.02219081 0.01755088 0.04916617 0.04137464 0.00657347 0.01317642
#  0.05128476 0.06427036 0.00433481 0.00370756 0.01332462 0.00482273
#  0.00962975 0.02958542 0.00347491 0.00427061 0.00749795 0.00417713
#  0.00506002 0.00511553 0.15684721 0.01564279 0.12547708 0.117608
#  0.00951396 0.0157771  0.02676024 0.15444403 0.0106431  0.00669794]

# model = GradientBoostingClassifier()

model =  XGBClassifier()

# 3. 훈련

model.fit(x_train, y_train)

# 4. 평가, 예측

acc = model.score(x_test, y_test)

# 5. 시각화

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
                align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importancs")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

print("acc :", acc)

print(model.feature_importances_)