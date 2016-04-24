from simple_features import *
import sklearn.neighbors as nn
import numpy as np
import quaternion

K = 1

model = nn.KNeighborsRegressor(K)
model.fit(train_features, train_params)

test_knn_indices = model.kneighbors(test_features, return_distance=False)
test_nns = [train_params[i[0]] for i in test_knn_indices]

test_distances = quaternion.distances(test_nns, test_params)
mean_error = np.mean(test_distances)

print(mean_error)
