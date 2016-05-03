import numpy as np
import tensorflow as tf

fingers = [
  'finger_index.01.R',
  'finger_index.02.R',
  'finger_index.03.R',

  'thumb.01.R.001',
  'thumb.01.R',
  'thumb.02.R',
  'thumb.03.R',

  'finger_middle.01.R',
  'finger_middle.02.R',
  'finger_middle.03.R',

  'finger_ring.01.R',
  'finger_ring.02.R',
  'finger_ring.03.R',

  'finger_pinky.01.R',
  'finger_pinky.02.R',
  'finger_pinky.03.R'
]

"""
xyzParams = [
  ('x', float),
  ('y', float),
  ('z', float),
]
"""

xyzParams = 3

cameraParams = [
  ('rotation', xyzParams),
  ('offset', xyzParams),
]

fingerParams = [('thumb.01.R.001', xyzParams)]
for i in [1, 2, 3]:
  fingerParams.append(('thumb.0%d.R' % i, xyzParams))

for finger in ['index', 'middle', 'ring', 'pinky']:
  for i in [1, 2, 3]:
    fingerParams.append(('finger_%s.0%d.R' % (finger, i), xyzParams))

sceneParams = [
  ('camera', cameraParams),
  ('fingers', fingerParams),
]

# camera rotated by
sceneParams2 = [('camera', 4)] + fingerParams

def size(schema):
  if isinstance(schema, int):
    return schema
  elif isinstance(schema, list):
    return sum([size(t) for _, t in schema])
  
  raise TypeError("Unknown schema %s" % schema)  

def flatten(schema, data):
  if isinstance(schema, int):
    return np.array(data)
  elif isinstance(schema, list):
    return np.concatenate([flatten(t, data[name]) for name, t in schema])
  
  raise TypeError("Unknown schema %s" % schema)

def unflatten(schema, flat_buffer):
  if isinstance(schema, int):
    return flat_buffer[:schema], flat_buffer[schema:]
  elif isinstance(schema, list):
    data = {}
    for name, t in schema:
      data[name], flat_buffer = unflatten(t, flat_buffer)
    return data
  
  raise TypeError("Unknown schema %s" % schema)

def placeholder(schema, shape, name):
  if isinstance(schema, int):
    return tf.placeholder(tf.float32, shape + [schema], name)
  elif isinstance(schema, list):
    placeholders = {}
    for n, t in schema:
      placeholders[n] = placeholder(t, shape, name + '/' + n)
    return placeholders
  
  raise TypeError("Unknown schema %s" % schema)

def feed(schema, name, values, feed_dict=None):
  if feed_dict is None:
    feed_dict = {}
  
  if isinstance(schema, int):
    feed_dict[name + ':0'] = np.array(values)
  elif isinstance(schema, list):
    for n, t in schema:
      feed(t, name + '/' + n, [v[n] for v in values], feed_dict)
  else:
    raise TypeError("Unknown schema %s" % schema)

  return feed_dict

def toList(schema, data, tensors=None):
  if tensors is None:
    tensors = []
  if isinstance(schema, int):
    tensors.append(data)
  elif isinstance(schema, list):
    for n, t in schema:
      toList(t, data[n], tensors)
  else:
    raise TypeError("Unknown schema %s" % schema)
  return tensors

def fromTensors(schema, tensors):
  if isinstance(schema, int):
    return tensors[0], tensors[1:]
  elif isinstance(schema, list):
    data = {}
    for n, t in schema:
      data[n], tensors = fromTensors(t, tensors)
    return data, tensors
  else:
    raise TypeError("Unknown schema %s" % schema)

def read(schema, tensors, sess, feed_dict):
  values = sess.run(toList(schema, tensors), feed_dict)
  values = zip(*values)
  return [fromTensors(schema, vals)[0] for vals in values]

