import bpy
import bmesh
import numpy
from mathutils import Vector, Color

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_blender_edit as lbe
import lib_blender_util as lbu
###########################################
scene = bpy.context.scene


body = bpy.data.objects["Suzanne"]#["Cube"]#


#trajectory = bpy.data.objects["trajectory"]
"""
trajectory = lbu.helix_nurbs(height=0.3,
                             radius=2.0,
                             npts=5,
                             name="helix")
"""
trajectory = bpy.data.objects["helix"]

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


nsteps = 5000
stat, coil = lbe.wrap_coil(body,
                           trajectory,
                           nsteps)

"""
# select mesh faces crossed by the coil
bpy.ops.object.select_all(action='DESELECT')
scene.objects.active = body
body.select = True
bpy.ops.object.mode_set(mode='EDIT')
bmsh = bmesh.from_edit_mesh(body.data)
bpy.ops.mesh.select_mode(type='FACE')
bpy.ops.mesh.select_all(action='DESELECT')
for p in coil:
    for f in p.faces:
        bmsh.faces[f].select = True
bpy.ops.object.mode_set(mode='OBJECT')
bmsh.free()
"""
thickness = body.dimensions.length*7e-3

obj = lbu.pydata_to_polyline([p.xyz for p in coil],
                             name='coil',
                             thickness=thickness,
                             resolution_u=24,
                             bevel_resolution=4,
                             fill_mode='FULL')
mat = bpy.data.materials.new("mat_coil")
mat.diffuse_color = [1,1,0]
mat.diffuse_intensity = 1
obj.data.materials.append(mat)

#lbe.set_smooth(body)
disp_value = lbe.apply_coil_pressure2(body,
                                      coil,
                                      thickness,
                                      smooth_passes=6,
                                      apply_disp=False)
disp_weight = numpy.asarray(disp_value)
disp_max = numpy.amin(disp_weight)
disp_weight = disp_weight/disp_max


# displacement value to weight (vertex groups)
scene.objects.active = body
bpy.ops.object.vertex_group_add()
vertgroup = body.vertex_groups[-1]
vertgroup.name = 'CoilPressure'
for i in range(len(disp_weight)):
    vertgroup.add([i], disp_weight[i], type='REPLACE')

# add 'displace' modifier
bpy.ops.object.modifier_add(type='DISPLACE')
dispmod = body.modifiers['Displace']
dispmod.vertex_group = 'CoilPressure'
#dispmod.mid_level = 0.0
dispmod.strength = disp_max


# displacement value to vertex color
print("displacement value --> vertex color")
disp_value = numpy.asarray(disp_value)
# remap to [0,1]
disp_min = numpy.amin(disp_value)
disp_max = numpy.amax(disp_value)
disp_rng = disp_max - disp_min
disp_value = (disp_value - disp_min)/disp_rng
body.data.vertex_colors.new()
vertexcolor = body.data.vertex_colors[0].data
for f in body.data.polygons:
    for i in range(f.loop_total):
        j = f.loop_start + i
        vertexcolor[j].color.h = 0
        vertexcolor[j].color.s = 0
        vertexcolor[j].color.v = disp_value[f.vertices[i]]

mat = bpy.data.materials.new("mat_body")
mat.use_vertex_color_paint = True
mat.use_shadeless = True
body.data.materials.append(mat)
body.active_material_index = 1


# unwrap UVs
print("displacement map")
scene = bpy.context.scene
bpy.ops.object.select_all(action='DESELECT')
body.select = True
scene.objects.active = body
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.smart_project(angle_limit=66.0,
                         island_margin=0.0,
                         user_area_weight=0.0)
bpy.ops.object.mode_set(mode='OBJECT')

render = scene.render
render.use_bake_clear = True
render.bake_margin = 5
render.use_bake_selected_to_active = True
render.bake_type = 'VERTEX_COLORS'

imsize = 1024
bpy.ops.image.new(name="displacement",
                  width=imsize,
                  height=imsize,
                  alpha=True,
                  generated_type='BLANK')
img = bpy.data.images["displacement"]
bpy.ops.object.mode_set(mode='EDIT')

area = bpy.context.screen.areas[2]
area.type = 'IMAGE_EDITOR'
area.spaces.active.image = img

pth = '/d/bandrieu/GitHub/Code/Python/Blender/'
filepath = pth + 'coil_pressure_displacement'
scene.render.filepath = filepath
bpy.ops.object.bake_image()
img.save_render(filepath=filepath + '.png')
bpy.ops.object.mode_set(mode='OBJECT')

# export obj
bpy.ops.export_scene.obj(filepath=pth + 'coil_body.obj',
                         use_selection=True,
                         use_animation=False,
                         use_mesh_modifiers=True,
                         use_edges=True,
                         use_smooth_groups=False,
                         use_smooth_groups_bitflags=False,
                         use_normals=False,
                         use_uvs=True,
                         use_materials=True,
                         use_triangles=False,
                         use_nurbs=False,
                         use_vertex_groups=False,
                         use_blen_objects=True,
                         group_by_object=False,
                         group_by_material=True,
                         keep_vertex_order=True,
                         axis_forward='-Z', axis_up='Y',
                         global_scale=1.0,
                         path_mode='AUTO')

