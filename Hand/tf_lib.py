import tensorflow as tf
import functools
import operator
import math

def squeeze_shape(shape):
  return list(filter(lambda x: x != 1, shape))

def flatten(x):
  size = product([dim.value for dim in x.get_shape()[1:]])
  return tf.reshape(x, [-1, size])

def leaky_relu(x, alpha=0.01):
  return tf.maximum(alpha * x, x)

def readImage(image_file):
  with tf.gfile.FastGFile(image_file, 'rb') as f:
    return f.read()

def product(xs):
  return functools.reduce(operator.mul, xs, 1)

def weight_variable(shape):
    '''
    Generates a TensorFlow Tensor. This Tensor gets initialized with values sampled from the truncated normal
    distribution. Its purpose will be to store model parameters.
    :param shape: The dimensions of the desired Tensor
    :return: The initialized Tensor
    '''
    #initial = tf.truncated_normal(shape, stddev=0.1)
    input_size = product(shape[:-1])
    initial = tf.truncated_normal(shape, stddev=1.0/math.sqrt(input_size))
    return tf.Variable(initial, 'weight')

def bias_variable(shape):
    '''
    Generates a TensorFlow Tensor. This Tensor gets initialized with values sampled from <some?> distribution.
    Its purpose will be to store bias values.
    :param shape: The dimensions of the desired Tensor
    :return: The initialized Tensor
    '''
    size = 1.0 / math.sqrt(product(shape))
    initial = tf.random_uniform(shape, -size, size)
    return tf.Variable(initial, 'bias')

def conv2d(x, W):
    '''
    Generates a conv2d TensorFlow Op. This Op flattens the weight matrix (filter) down to 2D, then "strides" across the
    input Tensor x, selecting windows/patches. For each little_patch, the Op performs a right multiply:
            W . little_patch
    and stores the result in the output layer of feature maps.
    :param x: a minibatch of images with dimensions [batch_size, height, width, 3]
    :param W: a "filter" with dimensions [window_height, window_width, input_channels, output_channels]
    e.g. for the first conv layer:
          input_channels = 3 (RGB)
          output_channels = number_of_desired_feature_maps
    :return: A TensorFlow Op that convolves the input x with the filter W.
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    Genarates a max-pool TensorFlow Op. This Op "strides" a window across the input x. In each window, the maximum value
    is selected and chosen to represent that region in the output Tensor. Hence the size/dimensionality of the problem
    is reduced.
    :param x: A Tensor with dimensions [batch_size, height, width, 3]
    :return: A TensorFlow Op that max-pools the input Tensor, x.
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convLayer(x, filter_size=5, filter_depth=64, pool_size=2):
  x_depth = x.get_shape()[-1].value
  W = weight_variable([filter_size, filter_size, x_depth, filter_depth])
  conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
  b = bias_variable([filter_depth])
  
  pool = tf.nn.max_pool(leaky_relu(conv+b),
                        ksize=[1,pool_size,pool_size,1],
                        strides=[1,pool_size,pool_size,1],
                        padding = 'SAME')
  
  return pool

def linearLayer(x, output_size):
  W = weight_variable([x.get_shape()[-1].value, output_size])
  return tf.matmul(x, W)

def affineLayer(x, output_size, nl=None):
  W = weight_variable([x.get_shape()[-1].value, output_size])
  b = bias_variable([output_size])
  
  fc = tf.matmul(x, W) + b
  
  return nl(fc) if nl else fc

# assumes that target is normalized
def quaternionDistance(predict, target):
  eps = 1e-8
  norm = tf.reduce_sum(tf.square(predict), 1)
  norm += eps
  #norm = tf.clip_by_value(norm, 1e-8, 100)
  dot = tf.reduce_sum(tf.mul(predict, target), 1)

  return tf.constant(1, tf.float32) - tf.square(dot) / norm

