import bpy
import bmesh

import numpy
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_blender_edit as lbe
import lib_blender_util as lbu
from mathutils import Vector
import random



scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

## add mesh
"""
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3)
obj = bpy.data.objects["Icosphere"]
"""


bpy.ops.mesh.primitive_monkey_add()
obj = bpy.data.objects["Suzanne"]
bpy.ops.object.modifier_add(type='SUBSURF')
obj.modifiers['Subsurf'].levels = 4
bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Subsurf')



"""
bpy.ops.mesh.primitive_cylinder_add(vertices=8,
                                    radius=1.0,
                                    depth=2.0,
                                    end_fill_type='TRIFAN')
obj = bpy.data.objects["Cylinder"]
bpy.ops.object.modifier_add(type='SUBSURF')
obj.modifiers['Subsurf'].levels = 4
bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Subsurf')
"""


"""
bpy.ops.import_scene.obj(filepath='/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/mesh_eos/mesh_eos_optim.obj',
                         axis_forward='Y', axis_up='Z')
obj = bpy.data.objects['mesh_eos_optim']
"""


"""
pth = '/d/bandrieu/GitHub/FFTsurf/cases/jouke/output/'
xyz = numpy.loadtxt(pth + 'pos_060.dat')
tri = numpy.loadtxt(pth + 'connect_01.dat')-1
verts = [[x for x in p] for p in xyz]
faces = [[int(v) for v in t] for t in tri]

obj = lbu.pydata_to_mesh(verts,
                         faces,
                         name='mesh')
"""

bpy.ops.object.select_all(action='DESELECT')
scene.objects.active = obj
obj.select = True

msh = obj.data



############################################

args = sys.argv
if len(args) < 4:
    length_max = 30.0
else:
    length_max = float(args[3])

if len(args) < 5:
    deviation = 0.0
else:
    deviation = float(args[4])


# random first point
iface = random.randint(1,len(obj.data.polygons))
baryco = numpy.random.rand(len(obj.data.polygons[iface].vertices))
startdirection = Vector(2*numpy.random.rand(3) - 1)

# compute geodesic
geodesic = lbe.trace_geodesic(iface,
                              baryco,
                              startdirection,
                              length_max,
                              deviation)


# select mesh faces crossed by the geodesic
bpy.ops.object.mode_set(mode='EDIT')
bmsh = bmesh.from_edit_mesh(msh)
bpy.ops.mesh.select_mode(type='FACE')
bpy.ops.mesh.select_all(action='DESELECT')
for p in geodesic:
    for f in p.faces:
        bmsh.faces[f].select = True
bpy.ops.object.mode_set(mode='OBJECT')
bmsh.free()


# trace the geodesic as a polyline
obj = lbu.pydata_to_polyline([p.xyz for p in geodesic],
                             name='geodesic',
                             thickness=obj.dimensions.length*1e-3,
                             resolution_u=24,
                             bevel_resolution=4,
                             fill_mode='FULL')
mat = bpy.data.materials.new("mat_geodesic")
mat.diffuse_color = [1,1,0]
mat.diffuse_intensity = 1
mat.emit = 1
mat.use_shadeless = True
obj.data.materials.append(mat)

