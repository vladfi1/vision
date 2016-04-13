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

def randomizeCamera(params=None):
  "Pick a random orientation to look at the hand"
  # default camera heading
  v = mathutils.Vector((0, 0, 1))
  v *= camera_distance
  
  q = randomQuaternion()
  m = q.to_matrix()
  
  camera.location = handR.head + m * v
  camera.rotation_mode = 'QUATERNION'
  camera.rotation_quaternion = q
  
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
  
  for i in range(1, 3):
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

def randomizeHand(params=None):
  if params == None:
    params = {}

  for bone in bonesR:
    theta = randomizeBone(bone)
    if theta is not None:
      params[bone.name] = list(theta)
  
  return params

def randomizeScene():
  params = randomizeHand()
  randomizeCamera(params)
  return params

bpy.context.scene.render.image_settings.file_format = 'JPEG'

def render(name):
  bpy.data.scenes['Scene'].render.filepath = '%s.jpeg' % name
  bpy.ops.render.render( write_still=True )

def genSimple(number):
  params = []

  for index in range(number):
    #params.append(list(randomizeCamera()))
    randomizeHand()
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

genTiered(10, 5)
