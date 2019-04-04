import bpy

import numpy

import random 

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_halfedge as lhe
import lib_blender_util as lbu

scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)


pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
iface = 2
strf = format(iface,'03')

"""
tri = numpy.loadtxt(pth + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pth + 'brepmesh/xyz_' + strf + '.dat', dtype=float)

verts = [[float(y) for y in x] for x in xyz]
faces = [[int(v) for v in t] for t in tri]


mesh = lhe.pydata_to_SurfaceMesh(xyz, tri)

obj = lbu.pydata_to_mesh(verts, faces)
me = obj.data
"""

bpy.ops.import_scene.obj(filepath='/d/bandrieu/stck/Bureau/PFE/MeshEdit/nefertitiobj.obj',
                         axis_forward='Y', axis_up='Z')
obj = bpy.data.objects['nefertitiobj']
me = obj.data

scene.objects.active = obj
obj.select = True

"""
bpy.ops.object.modifier_add(type='SUBSURF')
obj.modifiers['Subsurf'].subdivision_type = 'SIMPLE'
bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Subsurf')
bpy.ops.object.modifier_add(type='TRIANGULATE')
obj.modifiers['Triangulate'].quad_method = 'SHORTEST_DIAGONAL'
bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Triangulate')
"""


vco, f2v = lbu.mesh_to_pydata(me)
mesh = lhe.pydata_to_SurfaceMesh(vco, f2v)

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type='VERT')
bpy.ops.object.mode_set(mode='OBJECT')


boundary_loops = lhe.get_boundary_loops(mesh)

"""
# draw boundary loops
thickness = obj.dimensions.length*1e-3
for iloop, loop in enumerate(boundary_loops):
    lbu.pydata_to_polyline([me.vertices[ivert].co for ivert in loop],
                           name='boundary_loop_'+str(iloop),
                           thickness=thickness,
                           resolution_u=24,
                           bevel_resolution=4,
                           fill_mode='FULL')
"""

k = 1#boundary_loops[0][random.randint(0,len(boundary_loops[0]))]
#v = mesh.verts[random.randint(0,len(mesh.verts))]
v = mesh.verts[k]
me.vertices[v.index].select = True
"""
e = v.edge
while True:
    f = e[0]
    me.polygons[f].select = True
    e = lhe.get_prev(e, mesh)
    if lhe.is_boundary_edge(e, mesh): break
    e = lhe.get_twin(e, mesh)
    if e[0] == v.edge[0]: break
"""

ngb = lhe.get_v2v(mesh, v)
for w in ngb:
    me.vertices[w.index].select = True


scl = obj.dimensions.length*1e-2
me.vertices[v.index].co = me.vertices[v.index].co + scl*me.vertices[v.index].normal

"""
bpy.app.debug = True

e = v.edge
verts = []
for it in range(7):
    print('current:',e)
    print('face:', mesh.f2v[e[0]])
    w = mesh.verts[lhe.get_dest(e, mesh)]
    print('dest:',w.index)
    me.vertices[w.index].select = True
    verts.append(w)
    e = lhe.get_prev(e, mesh)
    print('prev:',e)
    if lhe.is_boundary_edge(e, mesh):
        w = mesh.verts[lhe.get_orig(e, mesh)]
        verts.append(w)
        break
    e = lhe.get_twin(e, mesh)
    print('twin:',e,'\n')
    if e[0] == v.edge[0]: break
"""
