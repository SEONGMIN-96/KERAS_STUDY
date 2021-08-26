# from tensorflow.keras.datasets import mnist
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

# OneHotEncoder

y_train = y_train[:,np.newaxis]
y_test = y_test[:,np.newaxis]

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

# 모델 구성

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# 히든 레이어

w = tf.Variable(tf.random_normal([28*28,512]), name='weight')
b = tf.Variable(tf.random_normal([512]), name='bias')

layers = tf.nn.relu(tf.matmul(x, w) + b)

# 히든 레이어

w = tf.Variable(tf.random_normal([512,256]), name='weight')
b = tf.Variable(tf.random_normal([256]), name='bias')

layers = tf.nn.relu(tf.matmul(layers, w) + b)

# 히든 레이어

w = tf.Variable(tf.random_normal([256,64]), name='weight')
b = tf.Variable(tf.random_normal([64]), name='bias')

layers = tf.nn.relu(tf.matmul(layers, w) + b)

# 아웃풋 레이어

w = tf.Variable(tf.random_normal([64,10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

# hyporthesis = x * w + b
# layers = tf.nn.relu(tf.matmul(x, w) + b)
# layers = tf.nn.elu(tf.matmul(x, w) + b)
# layers = tf.nn.selu(tf.matmul(x, w) + b)
# layers = tf.nn.dropout(layer, keep_prob=0.3)
# hypothesis = tf.sigmoid(tf.matmul(layers, w) + b)
hypothesis = tf.nn.softmax(tf.matmul(layers, w) + b)

# cost  = tf.reduce_mean(tf.square(hypothesis-y)) # mse
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy   
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))    # categorical_crossentropy

# optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(cost)

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(1001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0:
        print(epochs, "cost :", cost_val, '\n', hy_val)

predict = sess.run(hypothesis, feed_dict={x:x_test})
y_pred = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)

print('acc_score :', accuracy_score(y_test, y_pred))

sess.close()