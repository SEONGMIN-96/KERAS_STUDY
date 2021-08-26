from re import X
import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

w = tf.Variable(tf.random_normal([4,3]), name='weight')
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

pred, acc = session.run([prediction, accuracy], feed_dict={x:x_data, y:y_data})
print('prediction = ', pred, '\n', 'accuracy = ', acc)
#  accuracy =  0.37258348

session.close()

