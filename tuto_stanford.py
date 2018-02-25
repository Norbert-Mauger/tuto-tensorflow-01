import tensorflow as tf

tf.InteractiveSession()

a = tf.zeros((2,2)); b = tf.ones((2,2))

res = tf.reduce_sum(b, reduction_indices=1).eval()
print(res)
res = tf.reduce_sum(b).eval()
print(res)

print(a.get_shape())

a_vec = tf.reshape(a, (1,4))
print(a_vec.eval())


a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session() as sess:
    print(sess.run(c))
    print(c.eval())



state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

writer = tf.summary.FileWriter("./debug", sess.graph)
writer.close()
sess.close()