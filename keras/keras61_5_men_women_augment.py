# 훈련데이터를 기존데이터 20% 더 할것
# 완료후 기존 모델과 비교
# save_dir도 temp에 넣을 것
# 확인 후, 삭제

# 실습
# men women 데이터로 모델링을 구성할 것 !!

# 실습 2.
# 본인 사진으로 predict 하시오!!

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

# 1. ImageDataGenerator를 정의                      // x,y가 튜플 형태로 뭉쳐있음
# 2. 파일에서 땡겨오려면 -> flow_from_directory()   // x,y가 나눠있음
# 3. 데이터에서 땡겨오려면 -> flow()

augment_size = (x_data.shape[0] * 20 // 100)        # 추가할 데이터의 수를 맞춰준다.

randidx = np.random.randint(x_data.shape[0], size=augment_size)
print(x_data.shape[0])      # 3309
print(randidx)              # [39310 38997 11928 ... 40079 44541 58382]
print(randidx.shape)        # (681,)

x_augmented = x_data[randidx].copy()
y_augmented = y_data[randidx].copy()

x_data = np.concatenate((x_data, x_augmented))
y_data = np.concatenate((y_data, y_augmented))
print(x_data.shape)

# 데이터 증폭 후, 기존의 데이터에 더해준다.

# train_test_split 활용

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
        train_size=0.8, shuffle=True, random_state=66
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
model.add(Dense(512, activation='relu'))
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

after augment 
데이터가 커지면서 긍정적인 효과보임.
val_loss : 1.3622854948043823
val_acc : 0.6649873852729797
predict : [[0.23311535]]

'''