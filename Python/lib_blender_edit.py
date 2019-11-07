import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
import numpy
import math

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_blender_util as lbu


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
    subobj = lbu.pydata_to_mesh(verts, faces, name=obj.name + "_submesh")
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



### Laplacian smoothing ###
#"Improved Laplacian Smoothing of Noisy Surface Meshes", Vollmer et al. (2001)
def improved_laplacian_smoothing(obj,
                                 alpha=0.5,
                                 beta=0.5,
                                 npasses=1):
    #import numpy
    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    me = obj.data
    # Get a BMesh representation
    bm = bmesh.from_edit_mesh(me)
    nv = len(bm.verts)

    # get vertex-vertex adjacency lists
    neighbors = [[e.other_vert(v) for e in v.link_edges] for v in bm.verts]

    # perform plain Laplacian smoothing on free vertices
    o = [v.co.copy() for v in bm.verts]
    p = o[:]
    b = [Vector((0,0,0)) for v in bm.verts]
    for passe in range(npasses):
        q = p[:]#[v.co.copy() for v in bm.verts]
        for i, v in enumerate(bm.verts):
            if v.is_boundary: continue
            p[i] = Vector((0,0,0))
            n = len(neighbors[i])
            for w in neighbors[i]:
                p[i] += q[w.index]/n
            b[i] = p[i] - (alpha*o[i] + (1-alpha)*q[i])
        #
        for i, v in enumerate(bm.verts):
            if v.is_boundary: continue
            n = len(neighbors[i])
            for w in neighbors[i]:
                p[i] = p[i] - (1-beta)*b[w.index]/n
            p[i] = p[i] - beta*b[i]
        
    for i, v in enumerate(bm.verts):
        v.co[:3] = p[i][:3]
            
    # leave edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    bm.free()
    return
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




###
def set_smooth(obj):
    msh = obj.data
    msh.use_auto_smooth = True
    for f in msh.polygons:
        f.use_smooth = True
    return
###########################






###
def baryco_to_xyz(face, baryco):
    baryco = baryco/numpy.sum(baryco)
    x = [v.co for v in face.verts]
    xyz = Vector([0,0,0])
    for i in range(len(x)):
        xyz = xyz + x[i]*baryco[i]
    return xyz
###########################


###
def check_face_edge_crossing(orig,
                             disp,
                             face,
                             iedge,
                             EPS=1.e-7,
                             verbose=False):
    # edge vector
    vi = face.verts[iedge]
    vj = face.verts[(iedge+1)%len(face.verts)]
    vecij = vj.co - vi.co
    vecijsqr = vecij.dot(vecij)
    # edge wall := plane generated by edge ij and face normal
    planenormal = (vecij.cross(face.normal)).normalized()
    # 
    dest = orig + disp
    denom = planenormal.dot(disp)
    if abs(denom) < EPS:
        if verbose: print("        |denom| << 1")
        return False, Vector([0,0,0])
    else:
        fracdisp = planenormal.dot(vi.co - orig)/denom
        if verbose: print("        fracdisp =", fracdisp)
        if fracdisp < -EPS  or fracdisp > 1+EPS:
            return False, Vector([0,0,0])
        else:
            fracdisp = max(0., min(1.0, fracdisp))
            inter = orig + fracdisp*disp
            fracvij = vecij.dot(inter - vi.co)/vecijsqr#.length_squared
            if verbose: print("        fracvij =", fracvij)
            if fracvij < -EPS  or fracvij > 1+EPS:
                return False, Vector([0,0,0])
            else:
                fracvij = max(0., min(1.0, fracvij))
                return True, vi.co + fracvij*vecij
            


###
def trace_geodesic(iface,
                   baryco,
                   startdirection,
                   length_max=10.0,
                   deviation=0.0,
                   verbose=False):
    EPS = 1.e-7
    geodesic = []
    obj = bpy.context.active_object
    msh = obj.data
    
    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Get a BMesh representation
    bmsh = bmesh.from_edit_mesh(msh)

    # start point
    face = bmsh.faces[iface]
    xyz = baryco_to_xyz(face, baryco)
    if verbose: print("start point on face#", iface,
                      ", at barycentric co.=", baryco,
                      ", xyz co.=", xyz)

    # initialize geodesic
    geodesic.append(IntersectionPoint(faces=[face.index],
                                      uvs=[],
                                      xyz=xyz))
    direction = startdirection
    length = 0.0
    abort = False
    while True:
        if verbose: print("length=",length,"/",length_max)
        if verbose: print("  face #", face.index)
        normal = face.normal
        # apply random deviation to current direction
        if abs(deviation) > 0.0:
            random_direction = Vector(2*numpy.random.rand(3)-1).normalized()
            direction = direction + deviation*direction.length*random_direction
        # project displacement onto local tangent plane
        direction_t = (direction - direction.project(normal)).normalized()
        if verbose: print("  tangential direction =", direction_t)
        # scale by length residual
        displacement = (length_max - length)*direction_t
        #
        nv = len(face.verts)
        leaves_face = False
        for i in range(nv):
            if verbose: print("    ", i+1, "/", nv)
            if len(geodesic) > 1 and face.edges[i] == last_edge:
                continue
            leaves_face, xyznew = check_face_edge_crossing(geodesic[-1].xyz,
                                                           displacement,
                                                           face,
                                                           i,
                                                           EPS,
                                                           verbose)
            if verbose: print("        ", leaves_face)
            if leaves_face:
                edge = face.edges[i]
                delta_length = (xyznew - geodesic[-1].xyz).length
                if verbose: print("        delta_length =", delta_length)
                if delta_length < EPS:
                    leaves_face = False
                    break
                length += delta_length
                direction = geodesic[-1].xyz + displacement - xyznew
                if verbose: print("        new direction =", direction)
                geodesic.append(IntersectionPoint(faces=[face.index],
                                                  uvs=[],
                                                  xyz=xyznew))
                if edge.is_boundary:
                    print("hit boundary ---> abort")
                    abort = True
                    break
                else:
                    last_edge = edge
                    for otherface in edge.link_faces:
                        if otherface != face:
                            face = otherface
                            geodesic[-1].faces.append(face.index)
                            break
                    break

        if not leaves_face:
            print("does not leave face ---> abort")
            #geodesic.append(IntersectionPoint(faces=[],
            #                                  uvs=[],
            #                                  xyz=geodesic[-1].xyz + direction_t))
            abort = True
        if length >= length_max:
            print("length_max reached ---> abort")
            abort = True
            
        if abort:
            print("geodesic length =", length)
            # leave edit mode
            bpy.ops.object.mode_set(mode='OBJECT')
            bmsh.free()
            return geodesic
###########################




###
def wrap_coil(body,
              trajectory,
              nsteps):
    scene = bpy.context.scene
    coil = []
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
    location, contact_normal, iface = body.closest_point_on_mesh(lbu.get_global_position(emitter))
    if iface < 0:
        return False, coil
    contact_point = bpy.data.objects.new("contact_point", None)
    contact_point.location = location
    bpy.context.scene.objects.link(contact_point)

    # initialize wrapped coil
    coil.append(IntersectionPoint(faces=[iface],
                                  uvs=[],
                                  xyz=contact_point.location))

    # wrap coil around body
    EPS = body.dimensions.length*1e-5
    trajectory.data.path_duration = nsteps
    for step in range(nsteps+1):
        trajectory.data.eval_time = step
        scene.update()

        emitloc = lbu.get_global_position(emitter)
        location, normal, iface = body.ray_cast(start=emitloc,
                                                end=contact_point.location + EPS*contact_normal)

        if iface > 0:
            contact_point.location = location
            contact_normal = normal
            coil.append(IntersectionPoint(faces=[iface],
                                          uvs=[],
                                          xyz=location))

    return True, coil
###########################



def apply_coil_pressure(body,
                        coil,
                        thickness):
    bpy.ops.object.select_all(action='DESELECT')
    body.select = True
    bpy.context.scene.objects.active = body

    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Get a BMesh representation
    bmsh = bmesh.from_edit_mesh(body.data)
    
    bpy.ops.mesh.select_mode(type='FACE')

    for i, p in enumerate(coil):
        print("coil point #",i+1,"/",len(coil))
        bpy.ops.mesh.select_all(action='DESELECT')
        #select face(s) in direct contact with the current coil node
        for f in p.faces:
            bmsh.faces[f].select = True
        # extend selection
        for iextend in range(4):# HARD-CODED
            bpy.ops.mesh.select_more()
        # get list of vertices of the selected submesh
        faces = [f for f in bmsh.faces if f.select]
        verts = []
        for f in faces:
            for v in f.verts:
                if not v in verts:
                    verts.append(v)
        # move vertices
        for v in verts:
            vec = v.co - p.xyz
            C = vec.dot(vec) - thickness**2
            if C < 0:
                B = -v.normal.dot(vec)
                lamb = math.sqrt(B**2 - C) - B
                v.co = v.co - lamb*v.normal

    # leave edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bmsh.free()
    return
    
                
############################################
def apply_coil_pressure2(body,
                         coil,
                         thickness,
                         smooth_passes=3,
                         apply_disp=True):
    bpy.ops.object.select_all(action='DESELECT')
    body.select = True
    bpy.context.scene.objects.active = body

    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Get a BMesh representation
    bmsh = bmesh.from_edit_mesh(body.data)

    # compute displacement of vertices in direct contact with the coil
    nverts = len(bmsh.verts)
    vert_disp = numpy.zeros((nverts,3))
    nvert_disp = numpy.zeros(nverts, dtype=int)
    for i, p in enumerate(coil):
        print("coil point #",i+1,"/",len(coil))
        for f in p.faces:
            for v in bmsh.faces[f].verts:
                iv = v.index
                nvert_disp[iv] += 1
                vec = v.co - p.xyz
                C = vec.dot(vec) - thickness**2
                if C < 0:
                    B = -v.normal.dot(vec)
                    lamb = math.sqrt(B**2 - C) - B
                    disp = -lamb*v.normal
                    for i in range(3):
                        vert_disp[iv,i] = vert_disp[iv,i] + disp[i]
    for iv in range(len(bmsh.verts)):
        if nvert_disp[iv] > 0:
            vert_disp[iv] = vert_disp[iv]/float(nvert_disp[iv])

    # get list of free vertices (i.e. not in drect contact with the coil)
    freeverts = [v for v in bmsh.verts if nvert_disp[v.index] == 0]
    # get vertex-vertex adjacency lists
    neighbors = [[e.other_vert(v) for e in v.link_edges] for v in bmsh.verts]

    # smooth displacement field
    vert_disp_tmp = numpy.zeros((nverts,3))
    for passe in range(smooth_passes):
        print("displacement smoothing, pass #",passe+1,'/',smooth_passes)
        vert_disp_tmp[:,:] = 0.0
        for v in freeverts:
            iv = v.index
            vert_disp_tmp[iv] = vert_disp[iv].copy()
            for w in neighbors[v.index]:
                jv = w.index
                vert_disp_tmp[iv] = vert_disp_tmp[iv] + vert_disp[jv]
            vert_disp_tmp[iv] = vert_disp_tmp[iv]/float(len(neighbors[iv]) + 1)
        for v in freeverts:
            iv = v.index
            vert_disp[iv] = vert_disp_tmp[iv].copy()

    # extract normal displacement map
    disp_map = [v.normal.dot(Vector(vert_disp[v.index])) for v in bmsh.verts]

    if apply_disp:
        # apply vertex displacement
        for v in bmsh.verts:
            v.co = v.co + Vector(vert_disp[v.index])

    # leave edit mode
    bpy.ops.object.mode_set(mode='OBJECT')
    bmsh.free()
    return disp_map


###########################
def unwrap_uv_tensor_product(obj, u, v):
    uvlayer = obj.data.uv_layers.active
    if uvlayer is None:
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap(
            method='ANGLE_BASED',
            fill_holes=True,
            correct_aspect=True,
            use_subsurf_data=False,
            margin=0.001
        )
        bpy.ops.object.mode_set(mode='OBJECT')
        uvlayer = obj.data.uv_layers.active
    #
    m = len(u)
    n = len(v)
    for j in range(n-1):
        for i in range(m-1):
            k = i + j*(m-1)
            f = obj.data.polygons[k]
            for l in [0,3]:
                uvlayer.data[f.loop_start + l].uv[0] = u[i]
            for l in [1,2]:
                uvlayer.data[f.loop_start + l].uv[0] = u[i+1]
            for l in [0,1]:
                uvlayer.data[f.loop_start + l].uv[1] = v[j]
            for l in [2,3]:
                uvlayer.data[f.loop_start + l].uv[1] = v[j+1]
    return obj

###########################
def unwrap_uv_unstructured(obj, uv):
    uvlayer = obj.data.uv_layers.active
    if uvlayer is None:
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap(
            method='ANGLE_BASED',
            fill_holes=True,
            correct_aspect=True,
            use_subsurf_data=False,
            margin=0.001
        )
        bpy.ops.object.mode_set(mode='OBJECT')
        uvlayer = obj.data.uv_layers.active
    #
    for i, f in enumerate(obj.data.polygons):
        for j in range(f.loop_total):
            k = f.loop_start + j
            for l in range(2):
                uvlayer.data[k].uv[l] = uv[f.vertices[j]][l]
    return obj

###########################
#https://blender.stackexchange.com/questions/139384/get-rgb-value-of-texture-from-face-on-mesh/139399
#from mathutils.interpolate import poly_3d_calc
from bpy.types import Scene, Mesh, MeshPolygon, Image


# reference: https://svn.blender.org/svnroot/bf-extensions/trunk/py/scripts/addons/uv_bake_texture_to_vcols.py
def getPixel(img, uv_coord):
    """ get RGBA value for specified coordinate in UV image
    pixels    -- list of pixel data from UV texture image
    uv_coord  -- UV coordinate of desired pixel value
    """
    pixels = img.pixels # Accessing pixels directly is quite slow. Copy to new array and pass as an argument for massive performance-gain if you plan to run this function many times on the same image (img.pixels[:]).
    pixelNumber = (img.size[0] * int(uv_coord.y)) + int(uv_coord.x)
    return pixels[pixelNumber*4:(pixelNumber+1)*4]





def image_based_adaptive_subdivision(
        obj,
        image:Image,
        tolerance=0.01,
        smoothness=1,
        quadtri=True,
        quadcorner='PATH',
        n_pass_max=6
):
    bpy.context.scene.objects.active = obj
    msh = obj.data
    #
    for ipass in range(n_pass_max):
        listf = []
        for iface, face in enumerate(msh.polygons):
            if not face.select: continue
            # check if face needs to be further subdivided
            needs_to_be_further_subdivided = image_based_subdivision_criterion(
                mesh=msh,
                face=face,
                image=image,
                tolerance=tolerance
            )
            #
            if needs_to_be_further_subdivided:
                listf.append(iface)
        #
        # subdivide all selected faces
        nsubf = len(listf)
        if nsubf < 1:
            print('no more subdivisions needed')
            break
        # reset face selection
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')

        # select faces in list
        bpy.ops.object.mode_set(mode='OBJECT')
        for iface in listf:
            msh.polygons[iface].select = True
        
        # subdivide selected faces
        bpy.ops.object.mode_set(mode='EDIT')
        print('subdivide %d faces (pass #%d)' % (nsubf, ipass+1))
        bpy.ops.mesh.subdivide(smoothness=smoothness, quadtri=quadtri, quadcorner=quadcorner)
        bpy.ops.object.mode_set(mode='OBJECT')
    return 
   
def uv_in_image(uv, image):
    uv_loc = Vector((uv[0] % 1, uv[1] % 1))
    # convert uv_loc in range(0,1) to uv coordinate
    image_size_x, image_size_y = image.size
    x_co = round(uv_loc.x * (image_size_x - 1))
    y_co = round(uv_loc.y * (image_size_y - 1))
    return Vector((x_co, y_co))

def image_based_subdivision_criterion(mesh:Mesh, face:MeshPolygon, image:Image, tolerance):
    # get active uv layer data
    uv_layer = mesh.uv_layers.active
    assert uv_layer is not None # ensures mesh has a uv map
    uv = uv_layer.data
    # get uv coordinates of face's vertices
    uv_verts = [uv[i].uv for i in face.loop_indices]
    #uv_verts = [uv[face.loop_start+i].uv for i in range(face.loop_total)]
    invn = 1.0/len(uv_verts)
    uv_centroid = Vector((0,0))
    rgba_interp = numpy.zeros(4)
    for uvi in uv_verts:
        uv_centroid += uvi
        rgba_interp = rgba_interp + getPixel(image, uv_in_image(uvi, image))
    uv_centroid *= invn
    uv_centroid = uv_in_image(uv_centroid, image)
    
    rgba_interp = rgba_interp*invn
    #
    rgba_actual = numpy.asarray(getPixel(image, uv_centroid))
    #print('rgba interp, actual =', rgba_interp, rgba_actual)
    #
    diff_rgba = rgba_interp - rgba_actual
    diff_rgba = numpy.sqrt(numpy.sum(diff_rgba**2))
    #print('diff_rgba = ', diff_rgba, ' | ', tolerance)
    return diff_rgba > tolerance
    
