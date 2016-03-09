import pickle
import mxnet as mx

image_size = (256, 256)
param_size = 4

image = mx.symbol.Variable('image')
conv = mx.symbol.Convolution(image, kernel=(5, 5), num_filter=8)
relu = mx.symbol.Activation(conv, act_type='relu')
pool = mx.symbol.Pooling(relu, kernel=(8, 8), pool_type='max')
fc = mx.symbol.FullyConnected(data=pool, num_hidden=param_size, no_bias=True)
norm 

ground_truth = mx.symbol.Variable('ground_truth')
loss = 

model = mx.model.FeedForward(loss, 

