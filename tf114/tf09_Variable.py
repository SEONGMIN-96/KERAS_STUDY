import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import where_eager_fallback
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(where_eager_fallback)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa :", aaa)
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval()
print('bbb :', bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc :", ccc)
sess.close()