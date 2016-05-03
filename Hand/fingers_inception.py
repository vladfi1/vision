import tensorflow as tf
import tf_lib as tfl
import six
import pickle
import numpy as np
import inception
import scene

target = scene.placeholder(scene.sceneParams2, [None], 'target')

op = 'pool_1'
#features = tf.reshape(pool_3, [1, -1], name='features')
input_features = tf.placeholder(tf.float32, [None] + inception.getShape(op), name='features')

with tf.variable_scope('infer'):
  x = input_features
  for i, f in enumerate([128, 64, 32]):
    with tf.variable_scope('conv%d' % i):
      x = tfl.convLayer(x, filter_depth=f)
  
  flat = tfl.flatten(x)
  
  with tf.variable_scope('fc1'):
    fc1 = tfl.affineLayer(flat, 1024, tf.tanh)
  
  predict = {}
  for name, n in scene.fingerParams:
    with tf.variable_scope(name):
      predict[name] = tfl.affineLayer(fc1, n)
  
  with tf.variable_scope('camera'):
    predict['camera'] = tfl.affineLayer(fc1, 4, tf.tanh)

with tf.name_scope('loss'):
  loss_fingers = {}
  for name, _ in scene.fingerParams:
    with tf.name_scope(name):
      loss = tf.squared_difference(predict[name], target[name])
      loss = tf.reduce_sum(loss, 1)
      loss = tf.reduce_mean(loss)
      loss_fingers[name] = loss
  
  with tf.name_scope('fingers_total'):
    xs = [tf.reshape(x, [1]) for x in loss_fingers.values()]
    loss_fingers_total = tf.reduce_sum(tf.concat(0, xs))
  
  with tf.name_scope('camera'):
    loss_camera = tf.reduce_mean(tfl.quaternionDistance(predict['camera'], target['camera']))
  
  with tf.name_scope('total'):
    loss_total = loss_camera + loss_fingers_total

global_step = tf.Variable(0, name='global_step', trainable=False)
adamOpt = tf.train.AdamOptimizer(10.0 ** -4)
# train_q = opt.minimize(qLoss, global_step=global_step)
# opt = tf.train.GradientDescentOptimizer(0.0)
grads_and_vars = adamOpt.compute_gradients(loss_total)
trainAdam = adamOpt.apply_gradients(grads_and_vars, global_step=global_step)

#trainRMS = tf.train.RMSPropOptimizer(0.0001, momentum=0.1).minimize(loss_total)

data_dir = 'Data/Fingers/'

params_file = data_dir + 'params'
with open(params_file, 'rb') as f:
  params = pickle.load(f)

N = len(params)

features = inception.loadFeatures(data_dir, N, op)

data_size = N
#data_size = 200

#test_size = min(100, data_size // 10)
test_size = 100
train_size = data_size - test_size

batch_size = 5

train_queue = six.moves.queue.ModQueue(train_size // batch_size)

def feedBatch(index, train=True):
  index *= batch_size
  if train:
    index += test_size
  feed_dict = {input_features : features[index:index+batch_size]}
  scene.feed(scene.sceneParams2, 'target', params[index:index+batch_size], feed_dict)
  
  return feed_dict

def test():
  batches = test_size // batch_size
  
  predictions = []
  test_loss = 0.0

  for i in range(batches):
    feed_dict = feedBatch(i, False)
    predictions += scene.read(scene.sceneParams2, predict, sess, feed_dict)
    test_loss += sess.run(loss_total, feed_dict)
  
  return predictions, test_loss / batches

saver = tf.train.Saver(tf.all_variables())

sess = tf.Session()

def train():
  while True:
    for i in range(500):
      t, f, c = sess.run([trainAdam, loss_fingers_total, loss_camera], feedBatch(train_queue.dequeue()))
      print(f, c)
  
    predictions, test_loss = test()
    
    print("Test loss:", test_loss)
    
    with open(data_dir + 'predict', 'wb') as f:
      pickle.dump(predictions, f)
    
    saver.save(sess, 'Saves/fingers_inception')

def init():
  sess.run(tf.initialize_all_variables())

def restore():
  saver.restore(sess, 'Saves/fingers_inception')
