# from tensorflow.keras.datasets import mnist
from keras import datasets
from keras.datasets import mnist
from numpy.core.fromnumeric import shape
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

datasets = tf.keras.datasets.mnist

(x_train, x_test), (y_train, y_test) = datasets.load_data()

x_train = x_train.reshape(60000, 28*28*1)
x_test = x_test.reshape(10000, 28*28*1)

one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()

# 모델 구성

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

# 아웃풋 레이어

w = tf.Variable(tf.random_normal([28*28,10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

# hyporthesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost  = tf.reduce_mean(tf.square(hypothesis-y)) # mse
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy  
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))    # categorical_crossentropy 

# optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0:
        print(epochs, "cost :", cost_val, '\n', hy_val)

predicted = sess.run(hypothesis, feed_dict={x:x_test})
print(sess.run(tf.argmax(predicted, axis=1)))

sess.close()