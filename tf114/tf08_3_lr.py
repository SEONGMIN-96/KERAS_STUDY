# 실습
# tf08_2 파일의 lr을 수정해서
# epoch가 2000번이 아니라 100번 이하로 줄여라
# 결과치는
# step = 100 이하, w = 1.9999, b = 0.9999

import tensorflow as tf
tf.set_random_seed(66)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)  # 랜덤하게 내맘대로 넣어준
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32)  # 초기값

hypothesis = x_train * W + b
# f(x) = wx + b

predict = x_test * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.17338)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    _, loss_val, W_val, b_val, predict_val = sess.run([train, loss, W, b, predict],
                    feed_dict={x_train:[1,2,3], y_train:[3,5,7], x_test:[5,6]}
    )
    if step % 20 ==0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val, predict_val)

# predict 하는 코드를 추가하시오!!
# x_test라는 placeholder 생성!!