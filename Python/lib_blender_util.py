import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix



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
                   name='mesh'):
    msh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, msh)
    bpy.context.scene.objects.link(obj)
    msh.from_pydata(verts,[],faces)
    msh.update(calc_edges=True)
    return obj
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
