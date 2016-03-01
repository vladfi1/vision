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

def randomizeCamera():
  "Pick a random orientation to look at the hand"
  # default camera heading
  v = mathutils.Vector((0, 0, 1))
  v *= camera_distance
  
  q = randomQuaternion()
  m = q.to_matrix()
  
  camera.location = handR.head + m * v
  camera.rotation_mode = 'QUATERNION'
  camera.rotation_quaternion = q
  
  return q

def resetBone(bone):
  if bone.rotation_mode == 'QUATERNION':
    bone.rotation_quaternion.identity()
  else:
    bone.rotation_euler.zero()

def randomizeBone(bone):
  if bone.rotation_mode == 'QUATERNION':
    bone.rotation_quaternion = randomQuaternion().slerp(qid, 0.6)

for bone in bonesR:
  resetBone(bone)

#for bone in bonesR:
#  randomizeBone(bone)

#randomizeCamera()

def render(name):
  bpy.data.scenes['Scene'].render.filepath = '%s.png' % name
  bpy.ops.render.render( write_still=True )

def genSimple(number):
  params = []

  for index in range(number):
    params.append(list(randomizeCamera()))
    render(index)

  with open('params', 'wb') as params_file:
    pickle.dump(params, params_file)

genSimple(1000)

