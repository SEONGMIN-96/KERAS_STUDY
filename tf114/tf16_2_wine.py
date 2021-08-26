import imp
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
import numpy as np

datasets = load_wine()

x_data = datasets.data
y_data = datasets.target

# print(x_data.shape, y_data.shape)         # (178, 13) (178,)

# OneHotEncoder

y_data = y_data[:,np.newaxis]

enc = OneHotEncoder()
y_data = enc.fit_transform(y_data).toarray()

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([13,1]), name='weight')
b = tf.Variable(tf.random_normal([1,3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
prediction = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype=tf.float32))

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))    # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0e-9)
train = optimizer.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

for epochs in range(1001):
    cost_val, hypothesis_val, _ = session.run([cost, hypothesis, train], 
                                    feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, 'cost = ', cost_val, '\n',  hypothesis_val)

_, acc = session.run([prediction, accuracy], feed_dict={x:x_data, y:y_data})
print('accuracy = ', acc)

session.close()