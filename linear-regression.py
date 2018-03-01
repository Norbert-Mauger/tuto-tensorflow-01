# Inspired from this
# https://jasdeep06.github.io/posts/getting-started-with-tensorflow/
#
import numpy as np
import matplotlib
import tensorflow as tf
matplotlib.use('TkAgg')
import seaborn
import matplotlib.pyplot as plt


# Define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)


# Define data size and batch size
n_samples = 1000
batch_size = 100
# Tensorflow is finicky about shapes, so resize
X_data = np.reshape(X_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples,1))
# Define placeholders for input
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))
#placeholder for checking the validity of our model after training
X_check=tf.placeholder(tf.float32,shape=(n_samples,1))

# Define variables to be learned
with tf.variable_scope("linear-regression"):
    W = tf.get_variable("weights", (1, 1),
    initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1,),
    initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(X, W) + b
    loss = tf.reduce_sum((y - y_pred)**2/n_samples)

# Sample code to run full gradient descent:
# Define optimizer operation
opt_operation = tf.train.AdamOptimizer(.001).minimize(loss)
with tf.Session() as sess:
    # Initialize Variables in graph
    sess.run(tf.global_variables_initializer())
    # Gradient descent loop for 500 steps
    for _ in range(5000):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        # Do gradient descent step
        _, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch})

    # plotting the predictions
    y_check = tf.matmul(X_check, W) + b
    pred = sess.run(y_check, feed_dict={X_check: X_data})
    plt.scatter(X_data, pred)
    plt.scatter(X_data, y_data)
    plt.show()

