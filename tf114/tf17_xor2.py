# 인공지능의 겨울을 극복하자
# perceptron -> mlp

import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터

x_data = [[0,0], [0,1], [1,0], [1,1]]         # (4,2)
y_data = [[0], [1], [1], [0]]                 # (4,1)

# 모델 구성

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 히든 레이어

w = tf.Variable(tf.random_normal([2,8]), name='weight_1')
b = tf.Variable(tf.random_normal([8]), name='bias_1')

# hyporthesis = x * w + b
layer = tf.sigmoid(tf.matmul(x, w) + b)

#  행렬 연산은 tf.matmul사용하자

# 히든 레이어

w = tf.Variable(tf.random_normal([8,128]), name='weight_2')
b = tf.Variable(tf.random_normal([128]), name='bias_2')

# hyporthesis = x * w + b
layer = tf.sigmoid(tf.matmul(layer, w) + b)

#  행렬 연산은 tf.matmul사용하자

# 히든 레이어

w = tf.Variable(tf.random_normal([128,128]), name='weight_3')
b = tf.Variable(tf.random_normal([128]), name='bias_3')

# hyporthesis = x * w + b
layer = tf.sigmoid(tf.matmul(layer, w) + b)

#  행렬 연산은 tf.matmul사용하자

# 아웃풋 레이어

w = tf.Variable(tf.random_normal([128,1]), name='weight_4')
b = tf.Variable(tf.random_normal([1]), name='bias_4')

# hyporthesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(layer, w) + b)

# cost  = tf.reduce_mean(tf.square(hypothesis-y)) # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy   

# optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
            feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost :", cost_val, '\n', hy_val)

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print("예측값 :", hy_val, "\n 예측결과값 :", c, "\n 원래값 :", y_data, "\n Accuracy :", a)

sess.close()