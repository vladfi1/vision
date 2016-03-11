import tensorflow as tf

test_input = tf.placeholder(tf.float32, [2])
test_output = tf.reduce_sum(test_input)

with tf.Session() as sess:
  [out] = sess.run([test_output], feed_dict = {test_input : [1.0, 2.0]})
  print(out)
