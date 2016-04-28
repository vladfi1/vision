import bpy
import numpy
import mathutils
import pickle

# identity quaternion
qid = mathutils.Quaternion((1, 0, 0, 0))

def randomQuaternion():
  q = mathutils.Quaternion(numpy.random.randn(4))
  q.normalize()

  return q

camera_distance = 0.4
camera = bpy.context.scene.camera

#print(bpy.data.objects.items())

armature = bpy.data.objects['Armature']
bpy.context.scene.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

bones = armature.pose.bones

handR = bones['hand.R']

bonesR = [handR] + handR.children_recursive

def printBone(root):
  print(root.name, root.rotation_mode)
  for bone in root.children:
    printBone(bone)

#printBone(handR)

def setCamera(q):
  # default camera heading
  v = mathutils.Vector((0, 0, 1))
  v *= camera_distance
  
  m = q.to_matrix()
  
  camera.location = handR.head + m * v
  camera.rotation_mode = 'QUATERNION'
  camera.rotation_quaternion = q

def randomizeCamera(params=None, alpha=0.0):
  "Pick a random orientation to look at the hand"
  q = randomQuaternion()
  q = q.slerp(qid, alpha)
  setCamera(q)
  
  if params is not None:
    params['camera'] = list(q)
  
  return list(q)

def resetBone(bone):
  if bone.rotation_mode == 'QUATERNION':
    bone.rotation_quaternion.identity()
  else:
    bone.rotation_euler.zero()

def sampleTheta():
  return numpy.random.beta(1, 4)

def randomizeBone(bone):
  #bone.rotation_mode = 'QUATERNION'
  #bone.rotation_mode = 'XYZ'
  
  """
  if bone.rotation_mode == 'QUATERNION':
    q = randomQuaternion().slerp(qid, 0.85)
    bone.rotation_quaternion = q
    return q
  """
  
  if bone.name.startswith('finger'):
    bone.rotation_mode = 'XYZ'
    theta = sampleTheta()
    
    bone.rotation_euler.x = theta
    bone.rotation_euler.y = numpy.random.normal(0, 0.03)
    bone.rotation_euler.z = theta
    return bone.rotation_euler
  
  for i in range(1, 4):
    if bone.name == 'thumb.0%d.R' % i:
      bone.rotation_mode = 'XYZ'
      theta = sampleTheta()
      bone.rotation_euler.x = theta
      
      if i == 1:
        bone.rotation_euler.z = -theta
    
      return bone.rotation_euler
  
  if bone.name == 'thumb.01.R.001':
    bone.rotation_mode = 'XYZ'
    theta = sampleTheta()
    bone.rotation_euler.x = -theta
    
    return bone.rotation_euler
  
  return None

def resetScene():
  for bone in bonesR:
    resetBone(bone)

def randomizeHand(params=None, **kwargs):
  if params is None:
    params = {}

  for bone in bonesR:
    theta = randomizeBone(bone)
    if theta is not None:
      params[bone.name] = list(theta)
  
  return params

def randomizeScene(**kwargs):
  params = randomizeHand(**kwargs)
  randomizeCamera(params, **kwargs)
  return params

bpy.context.scene.render.image_settings.file_format = 'JPEG'

def render(name):
  bpy.data.scenes['Scene'].render.filepath = '%s.jpeg' % name
  bpy.ops.render.render( write_still=True )

# simple model, only one parameter, the camera pose
def genSimple(number):
  params = []

  for index in range(number):
    #params.append(list(randomizeCamera()))
    randomizeHand()
    render(index)

  with open('params', 'wb') as params_file:
    pickle.dump(params, params_file)

def renderSimple(params_file):
  with open(params_file, 'rb') as f:
    params = pickle.load(f)

  for i, q in enumerate(params):
    setCamera(q)
    render(i)

def genFingers(number, **kwargs):
  params = []

  for index in range(number):
    params.append(randomizeScene(**kwargs))
    render(index)

  with open('params', 'wb') as params_file:
    pickle.dump(params, params_file)

def genTiered(count, viewpoints):
  tiers = []
  for c in range(count):
    hand = randomizeHand()
    cameras = []
    for v in range(viewpoints):
      cameras.append(randomizeCamera())
      render("%d-%d" % (c, v))
    tiers.append((hand, cameras))
  
  with open('params', 'wb') as params_file:
    pickle.dump(tiers, params_file)

#genTiered(10, 5)
genFingers(10000, alpha=0.5)

