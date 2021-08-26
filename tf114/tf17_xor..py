import tensorflow as tf
tf.set_random_seed(66)

x_data = [[0,0], [0,1], [1,0], [1,1]]     # (4,2)
y_data = [[0], [1], [1], [0]]                 # (4,1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hyporthesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

#  행렬 연산은 tf.matmul사용하자

# cost  = tf.reduce_mean(tf.square(hypothesis-y)) # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy   

# optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer  = tf.train.GradientDescentOptimizer(learning_rate=1e-10)
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