import tensorflow as tf
import tf_lib as tfl
import os
import h5py
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

def getFeatures(image_file, ops='pool_3'):
  if isinstance(ops, basestring):
    tensors = op2Tensor(ops)
  else: # sequence of ops
    tensors = [op2Tensor(op) for op in ops]
  
  return sess.run(tensors, {image_str : tfl.readImage(image_file)})

def writeGraph(logdir='logs/'):
  summaryWriter = tf.train.SummaryWriter(logdir, sess.graph)
  summaryWriter.flush()

def cacheFeatures(image_dir, n, ops='pool_3'):
  if isinstance(ops, basestring):
    outputs = []
    for i in range(n):
      print(i)
      image_file = image_dir + str(i) + '.jpeg'
      outputs.append(getFeatures(image_file, ops))
    
    with open(image_dir + 'features', 'wb') as f:
      np.save(f, np.array(outputs))
  
  # TODO: multiple ops
