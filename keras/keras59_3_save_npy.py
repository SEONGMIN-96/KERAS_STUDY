import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    './_data/brain01_data/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=False,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    './_data/brain01_data/test',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=False,
)
# Found 120 images belonging to 2 classes.


# print(xy_train)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000028D21338550>
# # print(xy_train[0])
print(xy_train[0][0])       # x값
print(xy_train[0][1])       # y값
# # print(xy_train[0][2])     # 없음
print(xy_train[0][0].shape, xy_train[0][1].shape)
# (160, 150, 150, 3) (160,)
print(xy_test[0][0].shape, xy_test[0][1].shape)
# (120, 150, 150, 3) (120,)

# print(type(xy_train))         # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>  
# print(type(xy_train[0]))      # <class 'tuple'>
# print(type(xy_train[0][0]))   # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))   # <class 'numpy.ndarray'>

np.save('./_save/_npy/k59_3_x_train', arr=xy_train[0][0])
np.save('./_save/_npy/k59_3_x_test', arr=xy_test[0][0])
np.save('./_save/_npy/k59_3_y_train', arr=xy_train[0][1])
np.save('./_save/_npy/k59_3_y_test', arr=xy_test[0][1])