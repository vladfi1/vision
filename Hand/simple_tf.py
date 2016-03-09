import pickle
import tensorflow as tf
import tf_lib as tfl
from queue import *
import numpy as np
from scipy.ndimage import imread

width = 256
height = 256
channels = 3
param_size = 4
batch_size = 5

"""
image_input = tf.placeholder(tf.float32, shape=[width, height, channels])
target_input = tf.placeholder(tf.float32, shape=[param_size])

input_data = [image_input, target_input]
queue = tf.FIFOQueue(100, [d.dtype for d in input_data], shapes = [d.get_shape() for d in input_data])
enqueue_op = queue.enqueue(input_data)

image_batch, target_batch = queue.dequeue_many(batch_size)
"""
image_batch = tf.placeholder(tf.float32, shape=[None, width, height, channels])
target_batch = tf.placeholder(tf.float32, shape=[None, param_size])

channels = [64, 64, 32, 16]
conv = image_batch
for c in channels:
  conv = tfl.convLayer(conv, filter_depth=c)

shape = conv.get_shape()
flat_size = 1
for dim in shape[1:]:
  flat_size *= dim.value
print(flat_size)
conv_flat = tf.reshape(conv, [-1, flat_size])
#conv_flat = tf.reshape(conv, [batch_size, -1])

fc1 = tfl.affineLayer(conv_flat, 512, nl=tf.nn.tanh)
fc2 = tfl.affineLayer(fc1, param_size)

# quaternion distance
norms = tf.reduce_sum(tf.mul(fc2, fc2), 1)
dot = tf.reduce_sum(tf.mul(fc2, target_batch), 1)
gains = tf.mul(dot, dot) / norms
# losses = 1 - gains
loss = - tf.reduce_sum(gains)

trainGD = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#print(tf.all_variables())

def readPNG(filename):
  contents = tf.read_file(filename)
  image = tf.image.decode_png(contents)
  return tf.image.convert_image_dtype(image, tf.float32)

sess = tf.Session()

directory = 'Data/Simple/'

params = None
with open(directory + 'params', 'rb') as f:
  params = pickle.load(f)

data_size = len(params)

image_files = [directory + str(i) + '.png' for i in range(data_size)]
#filename_queue = tf.train.string_input_producer(filenames)

#validation = min(100, data_size // 10)
validation = 0
training = data_size - validation

def makeBatch(queue, size):
  indices = [queue.dequeue() for i in range(size)]

  images = [imread(image_files[i]) for i in indices]
  targets = [params[i] for i in indices]
  
  return {
    image_batch : np.array(images),
    target_batch : np.array(targets)
  }
"""
def validate():
  images = map(readPNG, image_files[-validation:])
  
  feed_dict = {
    image_batch : tf.pack(images),
"""


queue = ModQueue(validation)

sess.run(tf.initialize_all_variables())


for i in range(2):
  t, l = sess.run([trainGD, loss], makeBatch(queue, 2))
  print(l)
