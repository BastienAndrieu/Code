import bpy, bmesh
import math
from mathutils import Vector

scene = bpy.context.scene

obj_orig = bpy.data.objects['Suzanne']

# duplicate object
scene.objects.active = obj_orig
bpy.ops.object.duplicate_move()
obj = bpy.context.active_object
bpy.ops.object.duplicate_move()
obj_copy = bpy.context.active_object

# get the copy
scene.objects.active = obj

# switch to edit mode
bpy.ops.object.mode_set(mode='EDIT')

# get a BMesh representation
msh = obj.data

"""
bmsh = bmesh.new()
bmsh.from_mesh(msh)
"""
bmsh = bmesh.from_edit_mesh(msh)


# triangulate the BMesh
bmesh.ops.triangulate(bmsh,
                      faces=bmsh.faces[:],
                      quad_method=0,
                      ngon_method=0)

# apply Cotangent smoothing
lamb = 0.33
mu   = -0.34
nstep = 10

dv = []
for v in bmsh.verts:
    dv.append(Vector([0,0,0]))

edge_w = [0 for edge in bmsh.edges]
vert_sw = [0 for vert in bmsh.verts]

for step in range(nstep):
    for substep in range(2):
        if substep == 0:
            factor = lamb
        else:
            factor = mu
        
        # compute edge weights
        for i in range(len(bmsh.edges)):
            edge_w[i] = 0

        for face in bmsh.faces:
            for i, loop in enumerate(face.loops):
                angle = loop.calc_angle()
                cotan = 1.0/math.tan(angle)
                edge_w[face.edges[(i+1)%3].index] += cotan

        # compute dv
        for i in range(len(bmsh.verts)):
            vert_sw[i] = 0
            for j in range(3):
                dv[i][j] = 0

        for i, edge in enumerate(bmsh.edges):
            for vert in edge.verts:
                j = vert.index
                w = 0.5*edge_w[i]
                vert_sw[j] += w
                dv[j] = dv[j] + w*(edge.other_vert(vert).co - vert.co)

        # apply dv
        for i, v in enumerate(bmsh.verts):
            v.co = v.co + factor*dv[i]/vert_sw[i]

for i in range(len(bmsh.verts)):
    for j in range(3):
        obj_copy.data.vertices[i].co[j] = bmsh.verts[i].co[j]

bpy.ops.object.mode_set(mode='OBJECT')
bmsh.free()

"""
obj.hide = True
obj.hide_render = True
"""
bpy.ops.object.select_all(action='DESELECT')
obj.select = True
bpy.ops.object.delete(use_global=False)

# move copy to layer 2
obj_copy.layers[1] = True
obj_copy.layers[0] = False
