import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b

# 실습
# tf09 1번의 방식 3가지로 출력하시오!

sess = tf.Session()
sess.run(tf.global_variables_initializer())
hypothesis_1 = sess.run(hypothesis)
print("hypothesis_1 :", hypothesis_1)
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
hypothesis_2 = (x*W+b).eval()
print('hypothesis_2 :', hypothesis_2)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
hypothesis_3 = (x*W+b).eval(session=sess)
print("hypothesis_3 :", hypothesis_3)
sess.close()