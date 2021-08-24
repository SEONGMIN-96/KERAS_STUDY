# 실습, 과제
# keras61_5 남자 여자 데이터에 노이즈를 넣어
# 기미 주근깨 여드름 제거하시오!!

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random


# # 1. 데이터

person_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=0.1,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

x_data = np.load('./_save/_npy/k59_5_men_women_x.npy')
y_data = np.load('./_save/_npy/k59_5_men_women_y.npy')

datagen = ImageDataGenerator(
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

augment_size = (x_data.shape[0] * 20 // 100)        # 추가할 데이터의 수를 맞춰준다.

randidx = np.random.randint(x_data.shape[0], size=augment_size)
# print(x_data.shape[0])      # 3309
# print(randidx)              # [39310 38997 11928 ... 40079 44541 58382]
# print(randidx.shape)        # (681,)

x_augmented = x_data[randidx].copy()
y_augmented = y_data[randidx].copy()

x_data = np.concatenate((x_data, x_augmented))
y_data = np.concatenate((y_data, y_augmented))
# print(x_data.shape)

# 데이터 증폭 후, 기존의 데이터에 더해준다.

# train_test_split 활용

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
        train_size=0.8, shuffle=True, random_state=66
)

# 데이터 노이즈 추가

x_train_noised = x_train + np.random.normal(0, 0.2, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.2, size=x_test.shape)

print(x_train_noised.shape, x_train.shape)

# 2. 모델 구성

def autoencoder_deep(hidden_layer_size):
    model01 = Sequential()
    model01.add(Conv2D(filters=hidden_layer_size, kernel_size=(1,1), input_shape=(150, 150, 3)))
    model01.add(MaxPool2D())
    model01.add(Conv2D(filters=512, kernel_size=(1,1), activation='relu'))
    model01.add(UpSampling2D())
    model01.add(Conv2D(filters=3, kernel_size=(1,1), activation='sigmoid'))
    return model01

model = autoencoder_deep(hidden_layer_size=256)

# model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train_noised, x_train, epochs=50, batch_size=32, verbose=1,
          callbacks=[es], validation_split=0.2
)

# 4. 평가, 예측

output = model.predict(x_test_noised)

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
        plt.subplots(2  , 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다.

random_images = random.sample(range(output.shape[0]),5)

# 원본
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# noise cnn
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE_CNN", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()