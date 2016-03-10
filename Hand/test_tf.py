import tensorflow as tf

test_input = tf.placeholder(tf.float32, [4])
decoy = tf.Variable(tf.constant(1.0))
test_output = tf.reduce_sum(test_input) + decoy

trainGD = tf.train.GradientDescentOptimizer(0.01).minimize(test_output)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  [out] = sess.run([test_output], feed_dict = {test_input : 4 * [0.0]})
  print(out)
