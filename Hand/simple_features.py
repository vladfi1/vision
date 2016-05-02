import pickle
import numpy as np

data_dir = 'Data/Simple/'

params_file = data_dir + 'params'
with open(params_file, 'rb') as f:
  params = np.array(pickle.load(f))

features_file = data_dir + 'features'
with open(features_file, 'rb') as f:
  features = np.squeeze(np.load(f))

#data_size = len(params)
data_size = 200

#test_size = min(100, data_size // 10)
test_size = 100
test_params = params[:test_size]
test_features = features[:test_size]

train_params = params[test_size:data_size]
train_features = features[test_size:data_size]

