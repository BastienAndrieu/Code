import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix

import numpy

###
def clear_scene(meshes=True, lamps=True, cameras=False):
    scene = bpy.context.scene
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in scene.objects:
        if obj.type == 'MESH' and meshes:
            obj.select = True
        elif obj.type == 'LAMP' and lamps:
            obj.select = True
        elif obj.type == 'CAMERA' and cameras:
            obj.select = True

    bpy.ops.object.delete()
    return
###########################


###
def set_scene(resolution_x=800,
              resolution_y=600,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=10,
              use_environment_light=False,
              environment_energy=0.2,
              environment_color='PLAIN'):
    scene = bpy.context.scene
    
    render = scene.render
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    render.resolution_percentage = resolution_percentage
    render.alpha_mode = alpha_mode

    world = scene.world
    world.horizon_color = horizon_color

    light_settings = world.light_settings
    light_settings.samples = light_samples
    light_settings.use_environment_light = use_environment_light
    light_settings.environment_energy = environment_energy
    light_settings.environment_color = environment_color
    return
###########################


###
def get_global_position(obj):
    return Vector([obj.matrix_world[j][3] for j in range(3)])
###########################


### Insert a point light ###
def add_point_light(name="lamp",
                    energy=1,
                    shadow_method='RAY_SHADOW',
                    shadow_ray_samples=4,
                    shadow_soft_size=0,
                    location=[0,0,0]):
    scene = bpy.context.scene
    # Create new lamp datablock
    lamp_data = bpy.data.lamps.new(name=name, type='POINT')
    lamp_data.energy = energy
    lamp_data.shadow_method = shadow_method
    lamp_data.shadow_ray_samples = shadow_ray_samples
    lamp_data.shadow_soft_size = shadow_soft_size
    
    # Create new object with our lamp datablock
    lamp_object = bpy.data.objects.new(name=name, object_data=lamp_data)
    lamp_object.location = location

    # Link lamp object to the scene so it'll appear in this scene
    scene.objects.link(lamp_object)

    """
    # And finally select it make active
    lamp_object.select = True
    scene.objects.active = lamp_object
    """
    return lamp_object
###########################



### Insert mesh from pydata ###
def pydata_to_mesh(verts,
                   faces,
                   edges=None,
                   name='mesh'):
    if edges is None: edges = []
    msh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, msh)
    bpy.context.scene.objects.link(obj)
    msh.from_pydata(verts,edges,faces)
    if len(edges) < 1: msh.update(calc_edges=True)
    return obj
###########################

def mesh_to_pydata(mesh):
    verts = [[x for x in v.co[0:3]] for v in mesh.vertices]
    faces = [[i for i in f.vertices] for f in mesh.polygons]
    return verts, faces
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
        for j in range(len(p)):
            polyline.points[i].co[j] = float(p[j])
    polyline.points[i].co[3] = 1

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



###
def obb_to_mesh(center=[0,0,0], ranges=[1,1,1], axes=[[1,0,0],[0,1,0],[0,0,1]]):
    """
    center = Vector(center)
    verts = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                v = center +
                Vector(axes[0])*ranges[0]*(-1)**i +
                Vector(axes[1])*ranges[1]*(-1)**j +
                Vector(axes[2])*ranges[2]*(-1)**k
    faces = []
    """
    bpy.context.scene.objects.active = None
    bpy.ops.mesh.primitive_cube_add(location=center,
                                    rotation=Matrix(axes).to_euler())
    obj = bpy.context.active_object
    obj.scale = ranges
    return obj
###########################


### Test wether a point is directly visble from a camera ###
def is_visible_point(xyz,
                     cam=bpy.context.scene.camera,
                     tol=1.e-3,
                     nrs=32,
                     clean=True):
    bpy.ops.mesh.primitive_uv_sphere_add(location=xyz,
                                         size=tol,
                                         segments=nrs,
                                         ring_count=nrs)
    obj = bpy.context.active_object
    result = bpy.context.scene.ray_cast(start=cam.location,
                                        end=Vector(xyz))
    visible = (result[1] == obj)
    if clean:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.ops.object.delete()
    return visible
###########################



##
def helix_nurbs(height,
                radius,
                nturns,
                name="helix"):
    curv = bpy.data.curves.new(name=name,type='CURVE')
    curv.dimensions = '3D'

    obj = bpy.data.objects.new(name, curv)
    bpy.context.scene.objects.link(obj)

    nurbs = curv.splines.new('NURBS')
    
    npts = int(round(nturns*4.0))
    nurbs.points.add(npts-1)
    t = 2*numpy.pi*numpy.linspace(0,nturns,npts)
    z = numpy.linspace(0,height,npts)
    for i in range(npts):
        nurbs.points[i].co.x = radius*numpy.cos(t[i])
        nurbs.points[i].co.y = radius*numpy.sin(t[i])
        nurbs.points[i].co.z = z[i]
        nurbs.points[i].co.w = 1
    
    return obj
###########################

###########################
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_chebyshev as cheb
from lib_linalg import matmul

def bezier_surface_to_chebyshev(bsurf):
    spline = bsurf.data.splines[0]
    mb = spline.point_count_u #order_u
    nb = spline.point_count_v #order_v
    if max(mb,nb) > 4:
        return []
    B = numpy.zeros((mb,nb,3))
    for l, p in enumerate(spline.points):
        i = l%mb
        j = int((l-i)/mb)
        for k in range(3):
            B[i,j,k] = p.co[k]
    M = cheb.B2Cmatrix(mb)
    N = cheb.B2Cmatrix(nb)
    C = numpy.zeros((mb,nb,3))
    for dim in range(3):
        """
        for i in range(mb):
            for j in range(nb):
                for k in range(nb):
                    MB = 0.0
                    for l in range(mb):
                        MB += M[i,l]*B[l,k,dim]
                    C[i,j,dim] += MB*N[j,k]
        """
        C[:,:,dim] = matmul(matmul(M, B[:,:,dim]), N.T)
    return C
###########################



def tensor_product_mesh_vf(x, y, z, periodu=False, periodv=False):
    m = x.shape[0]
    n = x.shape[1]

    verts = []
    for j in range(n):
        for i in range(m):
            verts.append([x[i,j], y[i,j], z[i,j]])

    faces = []
    mf = m
    nf = n
    if not periodu:
        mf -= 1
    if not periodv:
        nf -= 1
    for j in range(nf):
        for i in range(mf):
            faces.append([j*m + i,
                          j*m + (i+1)%m,
                          ((j+1)%n)*m + (i+1)%m,
                          ((j+1)%n)*m + i])
    return verts, faces
