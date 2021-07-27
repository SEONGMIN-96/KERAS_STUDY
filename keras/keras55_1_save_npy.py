from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
from tensorflow.keras import datasets

#######################################################
datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

print(type(x_data), type(y_data))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data)
#######################################################

#######################################################
datasets = load_boston()

x_data = datasets.data
y_data = datasets.target

print(type(x_data), type(y_data))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data)
#######################################################

#######################################################
datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target

print(type(x_data), type(y_data))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_breaset_cancer.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_breaset_cancer.npy', arr=y_data)
#######################################################

#######################################################
datasets = load_diabetes()

x_data = datasets.data
y_data = datasets.target

print(type(x_data), type(y_data))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_diabetes.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_diabetes.npy', arr=y_data)
#######################################################

#######################################################
datasets = load_wine()

x_data = datasets.data
y_data = datasets.target

print(type(x_data), type(y_data))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data)
#######################################################

#######################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(type(x_train), type(y_train))
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save('./_save/_npy/k55_x_train_mnist.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_mnist.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_mnist.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_mnist.npy', arr=y_test)
#######################################################

#######################################################
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(type(x_train), type(y_train))
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save('./_save/_npy/k55_x_train_fashion.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_fashion.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_fashion.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_fashion.npy', arr=y_test)
#######################################################

#######################################################
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(type(x_train), type(y_train))
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save('./_save/_npy/k55_x_train_cifar10.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_cifar10.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_cifar10.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_cifar10.npy', arr=y_test)
#######################################################

#######################################################
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(type(x_train), type(y_train))
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np.save('./_save/_npy/k55_x_train_cifar100.npy', arr=x_train)
np.save('./_save/_npy/k55_y_train_cifar100.npy', arr=y_train)
np.save('./_save/_npy/k55_x_test_cifar100.npy', arr=x_test)
np.save('./_save/_npy/k55_y_test_cifar100.npy', arr=y_test)
#######################################################