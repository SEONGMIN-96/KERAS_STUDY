# 실습
# men women 데이터로 모델링을 구성할 것 !!

# 실습 2.
# 본인 사진으로 predict 하시오!!

from re import I, T
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers.pooling import MaxPooling2D

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

# test_datagen = ImageDataGenerator(rescale=1./255)

# men_women = person_datagen.flow_from_directory(
#     './_data/men_women01_data',
#     target_size=(200, 200),
#     batch_size=3309,
#     class_mode='binary',
#     shuffle=False,
# )
# # Found 3309 images belonging to 2 classes.

# print(men_women)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000152F0658550>
# print(type(men_women))
# # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(men_women[0][0]))
# # <class 'numpy.ndarray'>
# print(type(men_women[0]))
# # <class 'tuple'>
# print(men_women[0][0].shape)        # (3309, 150, 150, 3) m_w_x
# print(men_women[0][1].shape)        # (3309,)             m_w_y
# print(men_women[0][1])              # [0. 0. 0. ... 1. 1. 1.]

# # 이미지 변환이 시간이 오래걸리니, numpy파일을 저장한다.

# np.save('./_save/_npy/k59_5_men_women_x.npy', arr=men_women[0][0])
# np.save('./_save/_npy/k59_5_men_women_y.npy', arr=men_women[0][1])

# npy데이터를 로드해서 시간을 단축시킴
 
x_data = np.load('./_save/_npy/k59_5_men_women_x.npy')
y_data = np.load('./_save/_npy/k59_5_men_women_y.npy')

# train_test_split 활용

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
        train_size=0.8, shuffle=True, random_state=67
)

# 내 사진 데이터를 불러와서 수치화한다.

me_sample = person_datagen.flow_from_directory(
    './_data/men_women01_predict_data',
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary',
    shuffle=False,
)

print(me_sample[0][0].shape)    # (1, 150, 150, 3)
print(me_sample[0][1].shape)    # (1,)
print(me_sample[0][1])          # [0.]

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(150, 150, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

es = EarlyStopping(monitor='val_loss', mode='min', patience=10)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=1,
          callbacks=[es], validation_split=0.2
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)
pre = model.predict(me_sample[0][0])

print('val_loss :', loss[0])
print('val_acc :', loss[1])
print('predict :', pre)

'''

val_loss : 3.5088603496551514
val_acc : 0.5664652585983276
predict : [[0.01550011]]

val_loss : 1.4559051990509033
val_acc : 0.6027190089225769
predict : [[0.53685796]]

'''