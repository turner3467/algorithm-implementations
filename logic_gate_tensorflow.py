# Logic gate neural network implementation in tensorflow to compare to
# logic_gate_neural_net.R implementation.

import tensorflow as tf
import numpy as np

iterations = 4200
learningrate = 0.05
batch_size = 1
display_step = 100

X_mat = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])\
          .astype("float32")
X_mat_batch = np.split(X_mat, 4)
Y_mat = np.array([[0.0], [0.0], [0.0], [1.0]]).astype("float32")
Y_mat_batch = np.split(Y_mat, 4)

x = tf.placeholder("float32", [None, 2])
y = tf.placeholder("float32", [None, 1])


def logic_gate(x, weights):
    hidden_layer = tf.matmul(x, weights["h"])
    hidden_layer = tf.sigmoid(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights["o"])
    out_layer = tf.sigmoid(out_layer)
    return out_layer

weights = {"h": tf.Variable(tf.random_normal([2, 3])),
           "o": tf.Variable(tf.random_normal([3, 1]))}

pred = logic_gate(X_mat, weights)
cost = tf.sqrt(tf.reduce_sum(pred, Y_mat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)\
            .minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    print("Got here.")
    sess.run(init)

    for run in range(iterations):
        avg_cost = 0
        total_batch = 4

        for i in range(total_batch):
            batch_x, batch_y = X_mat_batch[i], Y_mat_batch[i]
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})

            avg_cost += c / total_batch
        if run % display_step == 0:
            print("Iteration: {}, cost: {}".format(run, avg_cost))
    print("Optimization finished!")

    feed_dict = {x: X_mat, y: Y_mat}
    print(sess.run(pred, feed_dict))
