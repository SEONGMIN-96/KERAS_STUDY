from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest',
)

# 1. ImageDataGenerator를 정의                      // x,y가 튜플 형태로 뭉쳐있음
# 2. 파일에서 땡겨오려면 -> flow_from_directory()   // x,y가 나눠있음
# 3. 데이터에서 땡겨오려면 -> flow()

augment_size = 10

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])     # 60000
print(randidx)              # [39310 38997 11928 ... 40079 44541 58382]
print(randidx.shape)        # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
# print(x_augmented.shape)    # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

import time

start_time = time.time()
x_augmented = train_datagen.flow(x_augmented,
                        np.zeros(augment_size),
                        batch_size=augment_size, shuffle=False,
                        save_to_dir='d:/temp',      # 요놈이 주인공!
).next()[0]
end_time = time.time() - start_time

print(x_augmented[0][0].shape)     # (40000, 28, 28, 1)

# flow() 는 iterator 
# x_augmented를 출력만해도 batch_size에 할당하는 img가 저장된다.
# .next()[0] .next() 차이



