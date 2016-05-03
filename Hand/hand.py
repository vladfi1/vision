import bpy
from numpy import random
import mathutils
import pickle
import math

# identity quaternion
qid = mathutils.Quaternion((1, 0, 0, 0))

def randomQuaternion():
  q = mathutils.Quaternion(random.randn(4))
  q.normalize()

  return q

def sampleTheta():
  return random.beta(1, 4)

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

def setCameraQ(q, camera_distance = 0.4):
  q = mathutils.Quaternion(q)
  q.normalize()
  
  # default camera heading
  v = mathutils.Vector((0, 0, camera_distance))
  
  m = q.to_matrix()
  
  camera.location = handR.head + m * v
  camera.rotation_mode = 'QUATERNION'
  camera.rotation_quaternion = q

def randomCameraOrientation():
  x = random.uniform(-1, 1)
  y = random.uniform(-1, 1)
  z = random.uniform(-1, 1)
  return [x, y, z]

def randomCameraOffset():
  z = random.uniform(0.4, 1.0)
  x = z * random.uniform(-0.3, 0.3)
  y = z * random.uniform(-0.3, 0.3)
  return [x, y, z]

def randomCamera():
  return {
    'rotation' : randomCameraOrientation(),
    'offset' : randomCameraOffset(),
  }

# use this orientation to face the hand
face_hand = mathutils.Euler((math.pi/2, 0, math.pi/2))

def setCamera(camera_params):
  v = mathutils.Vector(camera_params['offset'])
  camera.rotation_mode = 'XYZ'
  camera.rotation_euler = face_hand
  camera.rotation_euler.rotate(mathutils.Euler(camera_params['rotation']))
  camera.location = handR.head + camera.rotation_euler.to_matrix() * v

def randomFingerJoint(i):
  theta = sampleTheta()
  
  x = theta
  y = random.normal(0, 0.03)
  z = theta
  
  return [x, y, z]

def randomThumbJoint(i):
  theta = sampleTheta()
  x = theta
  y = 0.0
  z = -theta if i == 1 else 0.0
  
  return [x, y, z]

def randomThumbJoint001():
  theta = sampleTheta()
  x = -theta
  
  return [x, 0.0, 0.0]

def randomFingers():
  fingers = {}
  fingers['thumb.01.R.001'] = randomThumbJoint001()
  
  for i in [1, 2, 3]:
    fingers['thumb.0%d.R' % i] = randomThumbJoint(i)

  for finger in ['index', 'middle', 'ring', 'pinky']:
    for i in [1, 2, 3]:
      fingers['finger_%s.0%d.R' % (finger, i)] = randomFingerJoint(i)
  
  return fingers

def setFinger(name, value):
  bone = bones[name]
  bone.rotation_mode = 'XYZ'
  bone.rotation_euler = value

def setFingers(fingers):
  for name, value in fingers.items():
    setFinger(name, value)

def randomScene():
  return {
    'camera' : randomCamera(),
    'fingers' : randomFingers(),
  }

def setScene(scene):
  setCamera(scene['camera'])
  setFingers(scene['fingers'])

def setScene2(scene):
  for name, value in scene.items():
    if name == 'camera':
      setCameraQ(value)
    else:
      setFinger(name, value)

def resetBone(bone):
  if bone.rotation_mode == 'QUATERNION':
    bone.rotation_quaternion.identity()
  else:
    bone.rotation_euler.zero()

def resetScene():
  for bone in bonesR:
    resetBone(bone)

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

  for i, c in enumerate(params):
    setCamera(c)
    render(i)

def genScene(number):
  params = []

  for index in range(number):
    scene = randomScene()
    params.append(scene)
    setScene(scene)
    render(index)

  with open('params', 'wb') as params_file:
    pickle.dump(params, params_file)

def renderScene(params_file, name='%d'):
  with open(params_file, 'rb') as f:
    params = pickle.load(f)

  for i, s in enumerate(params):
    setScene(s)
    render(name % i)

def renderScene2(params_file, name='%d'):
  with open(params_file, 'rb') as f:
    params = pickle.load(f)

  for i, s in enumerate(params):
    setScene2(s)
    render(name % i)

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
#genFingers(10000)

#renderScene2('params')
renderScene2('predict', name='%d-predict')

