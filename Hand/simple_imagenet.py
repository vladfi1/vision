import tensorflow as tf
import tf_lib as tfl
import six
import pickle
import numpy as np

param_size = 4

with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name='')

sess = tf.Session()

image_str = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')

#writer = tf.train.SummaryWriter('./', sess.graph_def)
#target_batch = tf.placeholder(tf.float32, shape=[None, param_size])
target = tf.placeholder(tf.float32, shape=[param_size])

pool_3 = sess.graph.get_tensor_by_name('pool_3:0')

w = tf.truncated_normal([2048, param_size], stddev=0.1)
w = tf.Variable(w)

fc = tf.matmul(tf.reshape(pool_3, [-1, 2048]), w)
fc = tf.squeeze(fc)
fc = tf.tanh(fc)

#norms = tf.reduce_sum(tf.mul(fc, fc), 1)
#norms = tf.clip_by_value(norms, 1e-8, 100)
#dot = tf.reduce_sum(tf.mul(fc, target_batch), 1)

eps = 1e-8
norm = tf.reduce_sum(tf.mul(fc, fc))
#norm += eps
#norm = tf.clip_by_value(norm, 1e-8, 100)
dot = tf.reduce_sum(tf.mul(fc, target))

gain = tf.mul(dot, dot) / norm
loss = tf.constant(1, tf.float32) - gain

trainRMS = tf.train.RMSPropOptimizer(0.0001, momentum=0.1).minimize(loss)

directory = 'Data/Simple/'

params = None
with open(directory + 'params', 'rb') as f:
  params = pickle.load(f)

data_size = len(params)

image_files = [directory + str(i) + '.jpg' for i in range(data_size)]
#filename_queue = tf.train.string_input_producer(filenames)
data = zip(image_files, params)

validation = min(100, data_size // 10)
validation_files = image_files[-validation:]
validation_targets = params[-validation:]

training = data_size - validation
train_queue = six.moves.queue.ModQueue(training)

def readImage(image_file):
  return tf.gfile.FastGFile(image_file, 'rb').read()

def feedDict(i):
  return {
    image_str : readImage(image_files[i]),
    target : np.array(params[i])
  }

def validate():
  
  validation_loss = 0.0
  
  for i in range(data_size - validation, data_size):
    validation_loss += sess.run(loss, feedDict(i))
  
  return validation_loss / validation

saver = tf.train.Saver(tf.all_variables())

def train():
  counter = 0
  while True:
    for i in range(500):
      t, l, n = sess.run([trainRMS, loss, norm], feedDict(train_queue.dequeue()))
      print(l, n)
  
    print("Validation loss:", validate())
  
    saver.save(sess, 'Saves/simple_imagenet', global_step = counter)
  
    counter += 1

