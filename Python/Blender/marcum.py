import bpy, bmesh
import numpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_blender_util as lbu
import lib_halfedge as lhe

import math

#########################################################################################
def norm2(a):
    return numpy.sqrt( numpy.sum( numpy.power( a, 2 ) ) )
###########################################
def conformal_laplacian_matrix(obj):
    W = {}
    D = {}
    msh = obj.data
    bpy.ops.object.mode_set(mode='EDIT')
    # get a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(msh)
    # triangulate
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)
    for face in bm.faces:
        for i, loop in enumerate(face.loops):
            angle = loop.calc_angle()
            weight = max(1.0/math.tan(angle), 1e-2) # avoid degeneracy
            vjk = [face.verts[(i+1)%3].index, face.verts[(i+2)%3].index]
            h = lhe.hash_integer_pair(vjk[0], vjk[1])
            if h in W:
                W[h][2] += weight
            else:
                W[h] = [min(vjk[0], vjk[1]), max(vjk[0], vjk[1]), weight]
            for j in vjk:
                if j in D:
                    D[j] += weight
                else:
                    D[j] = weight
                    
    bm.free()
    bpy.ops.object.mode_set(mode='OBJECT')
    L = []
    for h in W:
        i = W[h][0]
        j = W[h][1]
        w = W[h][2]
        L.append([i,j,-w])
        L.append([j,i,-w])
    for i in D:
        L.append([i,i,D[i]])
    return L
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
lbu.clear_scene(meshes=True, lamps=False, cameras=False)
lbu.set_scene(resolution_x=800,
              resolution_y=800,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=10,
              use_environment_light=True,
              environment_energy=0.1,
              environment_color='PLAIN')

bpy.ops.import_scene.obj(filepath='/d/bandrieu/stck/Bureau/PFE/MeshEdit/nefertitiobj.obj',
                         axis_forward='Y', axis_up='Z')
cam = scene.camera
cam.location = [0,0,10]
cam.rotation_euler = [0,0,0]

obj = bpy.data.objects['nefertitiobj']
msh = obj.data

scene.objects.active = obj
obj.select = True
bpy.ops.view3d.camera_to_view_selected()


L = conformal_laplacian_matrix(obj)
f = open('/d/bandrieu/GitHub/Code/Python/Blender/conformal_laplacian_matrix.dat','w')
#for h in L:
#    f.write('%d %d %s\n' % (L[h][0], L[h][1], L[h][2]))
for rcw in L:
    f.write('%d %d %s\n' % (rcw[0], rcw[1], rcw[2]))
f.close()

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

f = open('/d/bandrieu/GitHub/Code/Python/Blender/f2v.dat','w')
for face in f2v:
    for i in face:
        f.write('%d ' % (i))
    f.write('\n')
f.close()


# get loops of boundary vertices
boundary_loops = lhe.get_boundary_loops(hmsh)

if len(boundary_loops) != 1:
    print("**** Need exactly 1 boundary loop, got ", len(boundary_loops), " ****")
    exit()

# extract boundary loop curve
boundary = boundary_loops[0]
f = open('/d/bandrieu/GitHub/Code/Python/Blender/boundary_vertices.dat','w')
for i in boundary:
    f.write('%d\n' % (i))
f.close()
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

# checker texture
imgchecker = bpy.data.images.load(filepath='/d/bandrieu/GitHub/Code/Python/Blender/checker.png')
texchecker = bpy.data.textures.new('texture_checker', 'IMAGE')
texchecker.image = imgchecker

# material
mat = bpy.data.materials.new('mat_checker')
mat.diffuse_color = [1,1,1]
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.specular_hardness = 30
mat.use_transparency = False
#mat.use_shadeless = True

slot = mat.texture_slots.add()
slot.texture = texchecker
slot.texture_coords = 'UV'
slot.blend_type = 'MULTIPLY'
slot.diffuse_color_factor = 0.33

obj.data.materials.append(mat)


# freestyle
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='EDGE')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')

scene.render.use_freestyle = True
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = False

lineset = freestyle.linesets['LineSet']
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = False
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.caps = 'BUTT'
linestyle.use_chaining = False
linestyle.geometry_modifiers['Sampling'].sampling = 0.1
linestyle.color = [0,0,0]
linestyle.thickness = 2.0
