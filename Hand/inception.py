import tensorflow as tf
import tf_lib as tfl

with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name='')

sess = tf.Session()

image_str = sess.graph.get_tensor_by_name('DecodeJpeg/contents:0')

pool_3 = sess.graph.get_tensor_by_name('pool_3:0')

def getFeatures(image_file):
  return sess.run(pool_3, {image_str : tfl.readImage(image_file)})

