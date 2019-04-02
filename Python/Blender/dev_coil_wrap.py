import bpy
import numpy
from mathutils import Vector

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_blender_edit as lbe
import lib_blender_util as lbu
###########################################
scene = bpy.context.scene


body = bpy.data.objects["Cube"]


trajectory = bpy.data.objects["trajectory"]

"""
# coil emitter
emitter = bpy.data.objects.new("emitter", None)
emitter.location = trajectory.data.splines[0].points[0].co[0:3]
bpy.context.scene.objects.link(emitter)
# "follow path" constraint
bpy.ops.object.select_all(action='DESELECT')
emitter.select = True
trajectory.select = True
bpy.context.scene.objects.active = trajectory #parent
bpy.ops.object.parent_set(type='FOLLOW') #follow path





# initial contact point
trajectory.data.eval_time = 0
location, contact_normal, index = body.closest_point_on_mesh(lbu.get_global_position(emitter))

contact_point = bpy.data.objects.new("contact_point", None)
contact_point.location = location
bpy.context.scene.objects.link(contact_point)

# initialize wrapped coil
coil = []
coil.append(contact_point.location)


# wrap coil around body
nsteps = 1000
trajectory.data.path_duration = nsteps
EPS = body.dimensions.length*1e-4
for step in range(nsteps+1):
    trajectory.data.eval_time = step
    scene.update()

    emitloc = lbu.get_global_position(emitter)
    location, normal, index = body.ray_cast(start=emitloc,
                                            end=contact_point.location + EPS*contact_normal)
    if index > 0:
        contact_point.location = location
        contact_normal = normal
        coil.append(location)
"""
nsteps = 1000
stat, coil = lbe.wrap_coil(body,
                           trajectory,
                           nsteps)

obj = lbu.pydata_to_polyline(coil,
                             name='coil',
                             thickness=body.dimensions.length*3e-3,
                             resolution_u=24,
                             bevel_resolution=4,
                             fill_mode='FULL')
mat = bpy.data.materials.new("mat_coil")
mat.diffuse_color = [1,1,0]
mat.diffuse_intensity = 1
mat.emit = 1
mat.use_shadeless = True
obj.data.materials.append(mat)
