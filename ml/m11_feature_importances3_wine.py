from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터

dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                    train_size=0.8, random_state=66
)

# 2. 모델

model = DecisionTreeClassifier()
# acc : 0.9444444444444444
# [0.         0.00489447 0.         0.0555874  0.05677108 0.
#  0.1569445  0.         0.         0.         0.03045446 0.33215293
#  0.36319516]

model = RandomForestClassifier()
# acc : 1.0
# [0.10503536 0.02598386 0.01388783 0.02662012 0.02931552 0.04756739
#  0.17229227 0.01295594 0.02403751 0.17308295 0.09723877 0.0941247
#  0.17785779]

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