import numpy as np
from sklearn.model_selection import train_test_split

x = np.array(range(1,101))

print(x)

x1, x2 = train_test_split(x, shuffle=False)

print(x1) # [ 1 ~ 75 ]
print(x2) # [ 76 ~ 100 ]

# train_size 의 default 값은 0.75이다.

from tensorflow.keras.layers import concatenate, Concatenate

print(type(concatenate)) # <class 'function'>
print(type(Concatenate())) # <class 'tensorflow.python.keras.layers.merge.Concatenate'>


