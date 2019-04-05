import bpy, bmesh
import numpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_blender_util as lbu
import lib_halfedge as lhe


#########################################################################################
def norm2(a):
    return numpy.sqrt( numpy.sum( numpy.power( a, 2 ) ) )
###########################################
def smoothing_closed_curve(p, n_passes=1):
    n = len(p)
    e = numpy.zeros(2)
    pnew = numpy.zeros(p.shape)
    q = numpy.zeros((3,p.shape[1]))
    for passe in range(n_passes):
        for i in range(n):
            k = (i+n-1)%n
            j = (i+1)%n
            eik = norm2(p[i] - p[k])
            eij = norm2(p[i] - p[j])
            ti = (eij - eik)/(eij + eik)
            pnew[i] = 0.5*ti*(ti - 1.0)*p[k] + (1.0 - ti**2)*p[i] + 0.5*ti*(ti + 1.0)*p[j]
        p = pnew.copy()
    return p
###########################################
def optimal_vertex_co_wrt_edge(vj, vk):
    vec = vk - vj
    vec90d = numpy.array([-vec[1], vec[0]])
    vi = 0.5*(vj + vk + numpy.sqrt(3.0)*vec90d)
    return vi
#########################################################################################
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

bpy.ops.import_scene.obj(filepath='/d/bandrieu/stck/Bureau/PFE/MeshEdit/nefertitiobj.obj',
                         axis_forward='Y', axis_up='Z')

obj = bpy.data.objects['nefertitiobj']
msh = obj.data

scene.objects.active = obj
obj.select = True

"""
# refine mesh
bpy.ops.object.modifier_add(type='SUBSURF')
obj.modifiers['Subsurf'].subdivision_type = 'SIMPLE'
bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Subsurf')
bpy.ops.object.modifier_add(type='TRIANGULATE')
obj.modifiers['Triangulate'].quad_method = 'SHORTEST_DIAGONAL'
bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Triangulate')
"""

# make halfedge representation
vco, f2v = lbu.mesh_to_pydata(msh)
hmsh = lhe.pydata_to_SurfaceMesh(vco, f2v)




# get loops of boundary vertices
boundary_loops = lhe.get_boundary_loops(hmsh)

if len(boundary_loops) != 1:
    print("**** Need exactly 1 boundary loop, got ", len(boundary_loops), " ****")
    exit()

# extract boundary loop curve
boundary = boundary_loops[0]
nb = len(boundary)



# get vertex-vertex adjacency lists
neighbors = [[w.index for w in lhe.get_v2v(hmsh, v)] for v in hmsh.verts]

# get list of interior vertices
interior = [v.index for v in hmsh.verts if not lhe.is_boundary_edge(v.edge, hmsh)]


n = len(msh.vertices)
uv = numpy.zeros((n,2))

# set initial uv boundary
t = numpy.linspace(0, 2*numpy.pi, nb+1)
uv[boundary,0] = numpy.cos(t[0:nb])
uv[boundary,1] = numpy.sin(t[0:nb])

for outerpasse in range(5):
    print("pass #", outerpasse)
    # apply combinatorial laplacian to interior nodes
    for innerpasse in range(100):
        uvtmp = uv.copy()
        for i in interior:
            for j in neighbors[i]:
                uvtmp[i] = uvtmp[i] + uv[j]
            uvtmp[i] = uvtmp[i]/float(len(neighbors[i]) + 1)
        duvmax = numpy.sqrt(numpy.amax(numpy.sum(numpy.power(uvtmp - uv,2), axis=1)))
        uv = uvtmp.copy()
        if duvmax < 1.e-3:
            break
    
    # "optimal" boundary nodes' coordinates
    for iv in boundary:
        uvopt = numpy.zeros(2)
        v = hmsh.verts[iv]
        nf = 0
        e = v.edge
        while True:
            f = e[0]
            i = e[1]
            j = hmsh.f2v[f][(i+1)%3]
            k = hmsh.f2v[f][(i+2)%3]
            uvopt = uvopt + optimal_vertex_co_wrt_edge(uv[j], uv[k])
            nf += 1
            #
            e = lhe.get_prev(e, hmsh)
            if lhe.is_boundary_edge(e, hmsh): break
            e = lhe.get_twin(e, hmsh)
        uv[iv] = uvopt/float(nf)
    
    # smooth boundary curve
    uv[boundary] = smoothing_closed_curve(uv[boundary], n_passes=10)

# leave edit mode
bpy.ops.object.mode_set(mode='OBJECT')

# rescale uv
uvmin = numpy.amin(uv, axis=0)
uvmax = numpy.amax(uv, axis=0)
uvctr = 0.5*(uvmin + uvmax)
uvrng = numpy.amax(0.5*(uvmax - uvmin))

uv = 0.5*(1 + (uv - uvctr)/uvrng)

# unwrap and apply uv coordinates
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.unwrap(method='ANGLE_BASED',
                  fill_holes=True,
                  correct_aspect=True,
                  use_subsurf_data=False,
                  margin=0.001)
bpy.ops.object.mode_set(mode='OBJECT')

for i, f in enumerate(msh.polygons):
    for j in range(f.loop_total):
        k = f.loop_start + j
        for l in range(2):
            obj.data.uv_layers.active.data[k].uv[l] = uv[f.vertices[j],l]
