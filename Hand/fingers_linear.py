import pickle
import numpy as np
from fingers import *
from sklearn.linear_model import LinearRegression

data_dir = 'Data/Fingers/'

params_file = data_dir + 'params'
with open(params_file, 'rb') as f:
  params = pickle.load(f)

features_file = data_dir + 'features'
with open(features_file, 'rb') as f:
  features = np.load(f)
features = np.squeeze(features)

data_size = len(params)

validation_size = min(100, data_size // 10)
training_size = data_size - validation_size

features_training = features[:training_size]
features_validation = features[-validation_size:]

def walkFingers(f):
  return {finger : [f(finger, i) for i in range(3)] for finger in fingers}

params_separated = walkFingers(lambda finger, i: [param[finger][i] for param in params])
params_training = walkFingers(lambda finger, i: params_separated[finger][i][:training_size])
params_validation = walkFingers(lambda finger, i: params_separated[finger][i][-validation_size:])

clfs = walkFingers(lambda finger, i: LinearRegression().fit(features_training, params_training[finger][i]))
scores = walkFingers(lambda finger, i: clfs[finger][i].score(features_validation, params_validation[finger][i]))

