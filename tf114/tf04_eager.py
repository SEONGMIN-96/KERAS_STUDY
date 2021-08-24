import tensorflow as tf
# print(tf.__version__) # 1.14.0

tf.compat.v1.disable_eager_execution()

# print('hello world') # tf1 버전은 기존의 방식 불가능

hello = tf.constant("Hello World")
# print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))

# b'Hello World' 