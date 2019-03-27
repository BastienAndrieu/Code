import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import numpy

### Insert mesh from pydata ###
def pydata_to_mesh(verts,
                   faces,
                   name='mesh'):
    msh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, msh)
    bpy.context.scene.objects.link(obj)
    msh.from_pydata(verts,[],faces)
    msh.update(calc_edges=True)
    return obj
###########################



### Triangulate an object ###
# to triangulate the active object, use:
# triangulate_object(bpy.context.active_object)
def triangulate_object(obj):
    me = obj.data
    # Get a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(me)

    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()
    return
###########################



### Submesh extraction ###
def extract_selected_submesh(obj):
    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    me = obj.data
    # Get a BMesh representation
    bm = bmesh.from_edit_mesh(me)

    # get list of selected faces
    listfaces = [f for f in bm.faces if f.select]

    # re-order vertices of the selected submesh and extract local connectivity
    verts = {} # dictionary: global vertex index -> local vertex index
    xyz = []   # xyz coordinates of the submesh's vertices
    faces = [] # connectivity table of the submesh's faces (local indices)
    for f in listfaces:
        fv = []
        for v in f.verts:
            if v.index not in verts:
                verts[v.index] = len(verts)
                xyz.append(list(v.co))
            fv.append(verts[v.index])
        faces.append(fv)

    # leave edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # clone submesh in a separate object
    subobj = pydata_to_mesh(verts, faces, name=obj.name + "_submesh")
    subobj.location = obj.location
    subobj.rotation_euler = obj.rotation_euler
    subobj.scale = obj.scale
    
    bm.free()
    return subobj
###########################



### Laplacian smoothing ###
def laplacian_smoothing(obj,
                        npasses=1):
    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    me = obj.data
    # Get a BMesh representation
    bm = bmesh.from_edit_mesh(me)

    # get list of free vertices (i.e. not on the boundary)
    freeverts = [v for v in bm.verts if not v.is_boundary]

    # get vertex-vertex adjacency lists
    neighbors = [[e.other_vert(v) for e in v.link_edges] for v in bm.verts]

    # perform plain Laplacian smoothing on free vertices
    for passe in range(npasses):
        for v in freeverts:
            cotmp = v.co.copy()
            for w in neighbors[v.index]:
                cotmp = cotmp + w.co
            cotmp = cotmp / float(len(neighbors[v.index]) + 1)
            v.co = cotmp
            
    # leave edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    bm.free()
    return
###########################



### Insert polyline from pydata  ###
def pydata_to_polyline(points,
                       name='polyline',
                       thickness=0,
                       resolution_u=24,
                       bevel_resolution=4,
                       fill_mode='FULL'):
    curv = bpy.data.curves.new(name=name,type='CURVE')
    curv.dimensions = '3D'

    obj = bpy.data.objects.new(name, curv)
    bpy.context.scene.objects.link(obj)

    polyline = curv.splines.new('POLY')
    polyline.points.add(len(points)-1)

    for i, p in enumerate(points):
        polyline.points[i].co = (p[0], p[1], p[2], 1)

    obj.data.resolution_u     = resolution_u
    obj.data.fill_mode        = fill_mode
    obj.data.bevel_depth      = thickness
    obj.data.bevel_resolution = bevel_resolution

    return obj
###########################



### Extract polyline vertex coordinates ###
def polyline_to_pydata(curv):
    polylines = []
    for polyline in curv.splines:
        points = numpy.zeros((len(polyline.points),3))
        for i, p in enumerate(polyline.points):
            for j in range(3):
                points[i,j] = p.co[j]
        polylines.append(points) 
    return polylines
###########################





### Intersection between a segment and a triangle in 3D ###
def segment_triangle_intersection(s0, s1,
                                  t0, t1, t2):
    mat = numpy.zeros((3,3))
    for i in range(3):
        mat[i,0] = t1[i] - t0[i]
        mat[i,1] = t2[i] - t0[i]
        mat[i,2] = s0[i] - s1[i]
        
    rhs = numpy.asarray(s0 - t0)

    try:
        uvw = numpy.linalg.solve(mat, rhs)
        #uvw, sing = solve_NxN(mat, rhs)

        result = numpy.amax(uvw) <= 1 and numpy.amin(uvw) >= 0 and uvw[0] + uvw[1] <= 1
        uvw = Vector(uvw)
        w = uvw.z
        xyz = (1. - w)*Vector(s0) + w*Vector(s1)
    except:
        # in case of singular matrix
        result = False
        uvw = Vector([0,0,0])
        xyz = Vector([0,0,0])
    return result, uvw, xyz
###########################


"""
def triangle_triangle_intersection(ta0, ta1, ta2,
                                   tb0, tb1, tb2):
    edge_tri_intersections = []
       
    for edge in [[ta0, ta1], [ta1, ta2], [ta2, ta0]]:
        nonempty, uvw, xyz = segment_triangle_intersection(edge[0], edge[1],
                                                           tb0, tb1, tb2)
        if nonempty:
            edge_tri_intersections.append(xyz)
    for edge in [[tb0, tb1], [tb1, tb2], [tb2, tb0]]:
        nonempty, uvw, xyz = segment_triangle_intersection(edge[0], edge[1],
                                                           ta0, ta1, ta2)
        if nonempty:
            edge_tri_intersections.append(xyz)
    return edge_tri_intersections
"""


############################################
class IntersectionPoint:
    def __init__(self, faces, uvs, xyz):
        self.faces = faces
        self.uvs = uvs
        self.xyz = xyz
        
    def print(self):
        print("faces =", self.faces, ", uvs =", self.uvs, ", xyz =", self.xyz)
############################################ 

### intersection of two triangles ###
def tri_tri_intersection(ta, tb):
    points = []
    for iedge in range(3):
        nonempty, uvw, xyz = segment_triangle_intersection(ta[iedge], ta[(iedge+1)%3],
                                                           tb[0], tb[1], tb[2])
        if nonempty:
            uvb = Vector(uvw[0:2])
            w = uvw.z
            if iedge == 0:
                uva = Vector([w, 0])
            elif iedge == 2:
                uva = Vector([0, 1. - w])
            else:
                uva = Vector([1. - w, w])
            points.append(IntersectionPoint(faces=[-1,-1], uvs=[uva, uvb], xyz=xyz))
    return points
###########################

### get Axis-Aligned Bounding Box of a set of points ###
def get_AABB(points):
    comin = points[0].copy()
    comax = points[0].copy()
    for p in points:
        for i in range(len(p)):
            comin[i] = min(comin[i], p[i])
            comax[i] = max(comax[i], p[i])
    return comin, comax
###########################

### test wether two Axis-Aligned Bounding Boxes overlap ####
def overlap_AABBs(amin, amax, bmin, bmax):
    for i in range(len(amin)):
        if max(amin[i], bmin[i]) > min(amax[i], bmax[i]): return False
    return True
###########################
