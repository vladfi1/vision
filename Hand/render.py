import bpy

#bpy.ops.scene.delete()

bpy.ops.scene.new(type='NEW')

bpy.ops.object.camera_add(location = (1, 1, 5), rotation = (0, 0, 0))
bpy.context.scene.camera = bpy.data.objects['Camera.001']

bpy.ops.mesh.primitive_cube_add(radius = 0.5)

"""
r = 10

for x in [-r, r]:
  for y in [-r, r]:
    for z in [-r, r]:
      bpy.ops.object.lamp_add(location = (x, y, z))
"""
bpy.ops.object.lamp_add(location = (3, 2, 1))

#print(bpy.data.objects.items())

bpy.data.scenes['Scene.001'].render.filepath = 'image.png'
bpy.ops.render.render( write_still=True )

