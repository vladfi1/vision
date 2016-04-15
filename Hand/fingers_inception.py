import tensorflow as tf
import tf_lib as tfl
import six
import pickle
import numpy as np
from fingers import *
from inception import *

finger_size = 3
camera_size = 4

with tf.variable_scope('target'):
  target_fingers = {finger : tf.placeholder(tf.float32, [finger_size], name=finger) for finger in fingers}
  #target_fingers_concat = tf.concat(target_fingers)
  target_camera = tf.placeholder(tf.float32, [camera_size], name='camera')
  
features = tf.reshape(pool_3, [1, -1], name='features')

with tf.variable_scope('predict'):
  with tf.variable_scope('fc1'):
    fc1 = tfl.affineLayer(features, 1024, tf.tanh)
  
  predict_fingers = {finger : tf.squeeze(tfl.affineLayer(fc1, finger_size), name=finger) for finger in fingers}
  
  with tf.variable_scope('camera'):
    predict_camera = tf.squeeze(tfl.affineLayer(fc1, camera_size, tf.tanh))

with tf.name_scope('loss'):
  loss_fingers = {finger : tf.squared_difference(predict_fingers[finger], target_fingers[finger], name=finger) for finger in fingers}
  
  loss_fingers_total = tf.reduce_sum(tf.concat(0, [loss_fingers[finger] for finger in fingers]))
  
  with tf.name_scope('camera'):
    loss_camera = tfl.quaternionDistance(predict_camera, target_camera)
  
  loss_total = loss_camera + loss_fingers_total

trainRMS = tf.train.RMSPropOptimizer(0.0001, momentum=0.1).minimize(loss_total)

directory = 'Data/Fingers/'

params = None
with open(directory + 'params', 'rb') as f:
  params = pickle.load(f)

data_size = len(params)

image_files = [directory + str(i) + '.jpeg' for i in range(data_size)]
data = zip(image_files, params)

validation = min(100, data_size // 10)
validation_files = image_files[-validation:]
validation_targets = params[-validation:]

training = data_size - validation
train_queue = six.moves.queue.ModQueue(training)

def feedDict(i):
  p = params[i]
  feed_dict = {target_fingers[finger] : np.array(p[finger]) for finger in fingers}
  feed_dict[target_camera] = np.array(p['camera'])
  feed_dict[image_str] = tfl.readImage(image_files[i])
  return feed_dict

def validate():
  validation_loss = 0.0
  
  for i in range(data_size - validation, data_size):
    validation_loss += sess.run(loss_total, feedDict(i))
  
  return validation_loss / validation

saver = tf.train.Saver(tf.all_variables())

def train():
  counter = 0
  while True:
    for i in range(500):
      t, f, c = sess.run([trainRMS, loss_fingers_total, loss_camera], feedDict(train_queue.dequeue()))
      print(f, c)
  
    print("Validation loss:", validate())
    
    saver.save(sess, 'Saves/fingers_inception', global_step = counter)
    
    counter += 1

