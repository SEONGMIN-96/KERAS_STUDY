# 가장 잘나온 전이학습모델로
# 이 데이터를 학습시켜서 결과치 도출
# keras59번과의 성능 비교

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import InceptionResNetV2

# 1. 데이터

# rps_datagen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=0.1,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest',
# )

# rps = rps_datagen.flow_from_directory(
#     './_data/rps',
#     target_size=(150, 150),
#     batch_size=2520,
#     class_mode='categorical',
#     shuffle=False,
# )
# # Found 2520 images belonging to 3 classes.

# print(rps)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000152F0658550>
# print(type(rps))
# # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(rps[0][0]))
# # <class 'numpy.ndarray'>
# print(type(rps[0]))
# # <class 'tuple'>
# print(rps[0][0].shape)        # (2520, 150, 150, 3) 
# print(rps[0][1].shape)        # (2520,)             
# print(rps[0][1])     

# # [[1. 0. 0.]                 
# #  [1. 0. 0.]
# #  [1. 0. 0.]
# #  ...
# #  [0. 0. 1.]
# #  [0. 0. 1.]
# #  [0. 0. 1.]]


# # 이미지 변환이 시간이 오래걸리니, numpy파일을 저장한다.

# np.save('./_save/_npy/k59_6_rps_x.npy', arr=rps[0][0])
# np.save('./_save/_npy/k59_6_rps_y.npy', arr=rps[0][1])

# npy데이터를 로드해서 시간을 단축시킴
 
x_data = np.load('./_save/_npy/k59_6_rps_x.npy')
y_data = np.load('./_save/_npy/k59_6_rps_y.npy')

# train_test_split 활용

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
        train_size=0.8, shuffle=True,
)

inceptionResNetV2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
inceptionResNetV2.trainable = True

# 2. 모델 구성

model = Sequential()

model.add(inceptionResNetV2)
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련

# 3_1. callbacks

es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1,
          callbacks=[es], validation_split=0.2
)

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('val_loss :', loss[0])
print('val_acc :', loss[1])

'''

기존 결과
val_loss : 0.77303546667099
val_acc : 0.7063491940498352

전의 학습 결과 (InceptionResNetV2)
val_loss : 0.8094532489776611
val_acc : 0.817460298538208

'''