import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
import warnings
from xgboost import XGBClassifier
import time

warnings.filterwarnings('ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

# to_categorical

x = x.reshape(70000, 28*28)

# pca = PCA(n_components=134)

# x = pca.fit_transform(x)

# pcr_EVR = pca.explained_variance_ratio_
# print(pcr_EVR.shape)

# cumsum = np.cumsum(pcr_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 0.94)+1)

# train_test

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV

x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size=0.8, shuffle=True, random_state=66
)

n_splits = 5
kflod = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
     "max_depth":[4,5,6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
     "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
     {"n_estimators":[90, 110], "learning_rate":[0.6, 0.9, 1],
     "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
      "colsample_bylevel":[0.6, 0.7, 0.9] }
]

n_jobs = -1

start_time = time.time()
model = GridSearchCV(XGBClassifier(), parameters, n_jobs=n_jobs, cv=kflod, verbose=1)
model.fit(x_train, y_train)

end_time = time.time() - start_time


print("최적의 매개변수 :", model.best_estimator_)
print("best_score_ :", model.best_score_)
print("model.score :", model.score(x_test, y_test))
print("소요시간 :", end_time)

