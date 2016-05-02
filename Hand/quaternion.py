import numpy as np

def distances(q1, q2):
  return 1 - np.square(np.sum(q1 * q2, axis=1))

def slerp(q1, q2, t=0.5):
  # TODO: implement
  pass
