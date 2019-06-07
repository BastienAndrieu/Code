import bpy, bmesh
from mathutils import Vector

scene = bpy.context.scene

obj_orig = bpy.data.objects['Suzanne']

# duplicate object
scene.objects.active = obj_orig
bpy.ops.object.duplicate_move()

# get the copy
obj = scene.objects.active

# switch to edit mode
bpy.ops.object.mode_set(mode='EDIT')

# get a BMesh representation
msh = obj.data
bmsh = bmesh.from_edit_mesh(msh)

# apply Taubin smoothing
lamb = 0.33
mu   = -0.34
nstep = 200

dv = []
for v in bmsh.verts:
    dv.append(Vector([0,0,0]))

for step in range(nstep):
    for substep in range(2):
        # compute dv
        for i, v in enumerate(bmsh.verts):
            for j in range(3):
                dv[i][j] = 0
            sum_w = 0
            for e in v.link_edges:
                w = 1 # weight for this adjecent vertex
                dv[i] = dv[i] + w*(e.other_vert(v).co - v.co)
                sum_w += w
            dv[i] = dv[i]/sum_w
            if substep == 0:
                dv[i] = dv[i]*lamb
            else:
                dv[i] = dv[i]*mu
        # apply smoothing substep
        for i, v in enumerate(bmsh.verts):
            v.co = v.co + dv[i]

# finish up, write the BMesh back to the mesh
bpy.ops.object.mode_set(mode='OBJECT')

# move copy to layer 2
obj.layers[1] = True
obj.layers[0] = False
