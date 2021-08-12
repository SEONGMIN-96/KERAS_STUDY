import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape)      # (70000, 28, 28)

# 실습
# pca를 통해 0.95 이상 가져가려면 몇개의 coulmns가 필요한가?

x = x.reshape(70000, 28*28)

# pca = PCA(n_components=28*28)
pca = PCA(n_components=154)

x = pca.fit_transform(x)

# pcr_EVR = pca.explained_variance_ratio_
# print("pcr_EVR :", pcr_EVR)

# cumsum = np.cumsum(pcr_EVR)
# print("pcr_EVR의 연속 합 :", cumsum)

# print("pcr_EVR 0.95를 위한 columns개수 :", np.argmax(cumsum >= 0.95)+1)