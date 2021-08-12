from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터

dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

model = DecisionTreeRegressor(max_depth=4)
# acc : 0.8774175457631728
# [0.03878833 0.         0.         0.         0.00802925 0.29639913
#  0.         0.05954596 0.         0.01862509 0.         0.
#  0.57861225]

# model = RandomForestRegressor()
# acc : 0.9226730640546915
# [0.03777992 0.00114303 0.00699198 0.00130014 0.02562012 0.42510804
#  0.01463562 0.06779926 0.00459551 0.01272414 0.01591433 0.01178495
#  0.37460295]

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