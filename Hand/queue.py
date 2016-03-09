
class ModQueue:
  def __init__(self, mod):
    self.mod = mod
    self.counter = 0
  
  def dequeue(self):
    c = self.counter
    self.counter = c + 1
    if self.counter == 0: self.counter = 0
    return c

