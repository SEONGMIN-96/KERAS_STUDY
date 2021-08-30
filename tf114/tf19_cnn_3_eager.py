from re import L
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())       # False
print(tf.__version__)               # 1.14.0 -> 2.4.1


# tf.set_random_seed(66)

# 1. 데이터

from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

tf.compat.v1.set_random_seed(55)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

learning_rate = 0e-2
training_epochs = 10
batch_size = 2
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None,32, 32, 3])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 모델구성

w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 3, 32]) # shape = [kernel_size, chanel(input), filter(output)]
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME'
)

# model = Sequential
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
#           padding='same', input_shape=(28, 28, 1)    
# ))
# model.add(MaxPool2D())

# print(w1)   # (3, 3, 3, 32)
# print(L1)   # (?, 32, 32, 32)
# print(L1_maxpool)   # (?, 16, 16, 32)

w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME'
)

# print(L2)   # (?, 16, 16, 64)
# print(L2_maxpool)   # (?, 8, 8, 64)

w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME'
)

# print(L3)   # (?, 8, 8, 128)
# print(L3_maxpool)   # (?, 4, 4, 128)

w4 = tf.compat.v1.get_variable('w4', shape=[3, 3, 128, 64])#, 
                    # initializer=tf.contrib.layers.xavier_initializer()
# )
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME'
)

# print(L4)   # (?, 4, 4, 64)
# print(L4_maxpool)   # (?, 2, 2, 64)

# Flatten

L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64])
# print("flatten :", L_flat)      # (?, 256)

w5 = tf.compat.v1.get_variable("w5", shape=[2*2*64, 128])#,
#                     initializer=tf.contrib.layers.xavier_initializer()
# )
b5 = tf.Variable(tf.random.normal([128]), name='b1')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)

# print(L5)       # (?, 64)

# Layer DNN

w6 = tf.compat.v1.get_variable('w6', shape=[128, 32])
b6 = tf.Variable(tf.random.normal([32]), name='b2')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)

# print(L6)       # (?, 32)

# Layer Softmax

w7 = tf.compat.v1.get_variable("w7", shape=[32, 10])
b7 = tf.Variable(tf.random.normal([10]), name='b7')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)

# print(hypothesis)   # (?, 10)

# 3. 컴파일 훈련

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))    # categorical_crossentropy
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0e-9).minimize(loss)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# total_batch = int(len(x_train/batch_size))

for epoch in range(training_epochs):

    avg_loss = 0 

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch

    print('Epoch :', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))

print("훈련 끝!!")

prediction = tf.equal(tf.compat.v1.argmax(hypothesis, 1), tf.compat.v1.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print("ACC :", sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# batch_normalizer, dropout처럼 과적합을 방지함.
# 가중치를 규제, 초기화역시 가능, 0으로 초기화하진 않음 (bias 가능)
# 가중치를 초기화할시 캐바캐지만 통상적으로 좋음.
# 
# '''