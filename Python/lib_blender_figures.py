import bpy
import bpy_extras
from mathutils import Vector

### get coordinates in image space of xyz point
def convert_3d_to_2d_coords(xyz, normalize=True):
    scene = bpy.context.scene
    cam = scene.camera
    
    render_scale = scene.render.resolution_percentage / 100
    render_w = int(scene.render.resolution_x * render_scale)
    render_h = int(scene.render.resolution_y * render_scale)

    co = Vector(xyz)
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
    u = co_2d.x - 2.*cam.data.shift_x
    v = co_2d.y - 2.*float(render_w)/float(render_h)*cam.data.shift_y

    if not normalize:
        u *= render_w
        v *= render_h
    
    return u, v
###############################

# /!\ works only in 'PERSPECTIVE' mode ('ORTHO' not yet supported)
def fit_camera_to_meshes(meshes):
    scn = bpy.context.scene
    cam = scn.camera
    ratio = scn.render.resolution_x/scn.render.resolution_y
    shx = cam.data.shift_x
    shy = cam.data.shift_y
    ### compute original bounding box in image space
    xmin = 1
    xmax = 0
    ymin = 1
    ymax = 0
    for me in meshes:
        for v in me.vertices:
            co_obj = v.co
            co_img = bpy_extras.object_utils.world_to_camera_view(scn, cam, co_obj)
            x = co_img.x - shx
            y = co_img.y - shy*ratio
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
    xctr = 0.5*(xmin + xmax)
    xrng = 0.5*(xmax - xmin)
    yctr = 0.5*(ymin + ymax)
    yrng = 0.5*(ymax - ymin)
    ### adjust camera sensor size to fit 
    scl = 0.5/max(xrng, yrng)
    cam.data.sensor_width /= scl
    ### adjust camera shift to align the center of the view with
    #   the center of the bounding box
    cam.data.shift_x = scl*(xctr - 0.5)
    cam.data.shift_y = scl*(yctr - 0.5)/ratio
    return
###############################
