import tensorflow as tf
import tf_lib as tfl
import os
#import h5py
import numpy as np

with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name='')

sess = tf.Session()

image_str = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')

#pool_3 = sess.graph.get_tensor_by_name('pool_3:0')

op_names = [op.name for op in sess.graph.get_operations()]

mixed_suffixes = [''] + ['_' + str(i) for i in range(1, 11)]
mixed_joins = ['mixed' + s + '/join' for s in mixed_suffixes]
layers = ['conv', 'conv_1', 'conv_2', 'pool', 'conv_3', 'conv_4', 'pool_1'] + mixed_joins + ['pool_3']

def op2Tensor(op):
  "Assumes that operations have one output."
  return op + ':0'

def getTensor(op):
  return sess.graph.get_tensor_by_name(op2Tensor(op))

def getShape(op):
  return tfl.squeeze_shape([dim.value for dim in getTensor(op).get_shape()])

def getFeatures(image_file, ops='pool_3'):
  if isinstance(ops, str):
    tensors = op2Tensor(ops)
  else: # sequence of ops
    tensors = [op2Tensor(op) for op in ops]
  
  return sess.run(tensors, {image_str : tfl.readImage(image_file)})

def writeGraph(logdir='logs/'):
  summaryWriter = tf.train.SummaryWriter(logdir, sess.graph)
  summaryWriter.flush()

def cacheFeatures(image_dir, n, ops='pool_3'):
  if isinstance(ops, str):
    ops = [ops]
  
  files = [np.memmap(filename=image_dir + op, mode='w+', dtype=np.float32, shape=tuple([n]+getShape(op))) for op in ops]
  
  for i in range(n):
    print(i)
    image_file = image_dir + str(i) + '.jpeg'
    output = getFeatures(image_file, ops)
    for f, x in zip(files, output):
      f[i] = np.squeeze(x)

def loadFeatures(path, n, op):
  return np.memmap(filename=path+op, mode='r', dtype=np.float32, shape=tuple([n]+getShape(op)))

def cacheTiered(image_dir, count, viewpoints, ops='pool_3'):
  if isinstance(ops, str):
    hands = []
    for c in range(count):
      print(c)
      cameras = []
      for v in range(viewpoints):
        image_file = image_dir + '%d-%d.jpeg' % (c, v)
        cameras.append(getFeatures(image_file, ops))
      hands.append(cameras)
    
    with open(image_dir + 'features', 'wb') as f:
      np.save(f, np.array(hands))
  
  # TODO: multiple ops

