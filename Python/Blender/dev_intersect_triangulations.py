import bpy
import numpy
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_blender_edit as lbe
from mathutils import Vector

import os



pth = '/d/bandrieu/GitHub/Code/Python/Blender/'

############################################
scene = bpy.context.scene


# clear scene
for ob in scene.objects:
    if ob.type == 'CAMERA':
        ob.select = False
    else: 
        ob.select = True
        
bpy.ops.object.delete()

# add cube
#bpy.ops.mesh.primitive_cube_add()


## add sphere
bpy.ops.mesh.primitive_ico_sphere_add(location=[-0.84, -0.58, 1.07], subdivisions=4)

# add monkey
bpy.ops.mesh.primitive_monkey_add(location=[-0.84, -0.58, 1.07])
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Subsurf')

############################################


#obj = [bpy.data.objects["Cube"], bpy.data.objects["Icosphere"]]
#obj = [bpy.data.objects["Cube"], bpy.data.objects["Suzanne"]]
obj = [bpy.data.objects["Icosphere"], bpy.data.objects["Suzanne"]]

# set camera
for o in obj:
    o.select = True
bpy.ops.view3d.camera_to_view_selected()
for o in obj:
    o.select = False

for imesh, o in enumerate(obj):
    # apply transformation
    bpy.context.scene.objects.active = o
    o.select = True
    bpy.ops.object.transform_apply(location=True, rotation=True)
    
    # triangulate
    lbe.triangulate_object(o)
    
    # unwrap UVs
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66.0,
                             island_margin=0.0,
                             user_area_weight=0.0)
    bpy.ops.uv.export_layout(filepath=pth+'uvlayout_'+str(imesh),
                             export_all=False,
                             modified=False,
                             mode='PNG',
                             size=(1024, 1024),
                             opacity=1,
                             tessellated=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    o.select = False




####
segments = []
msh = [o.data for o in obj]

print('Computing intersections...')
for fa in msh[0].polygons:
    ta = [msh[0].vertices[v].co for v in fa.vertices]
    amin, amax = lbe.get_AABB(ta)
    for fb in msh[1].polygons:
        tb = [msh[1].vertices[v].co for v in fb.vertices]
        bmin, bmax = lbe.get_AABB(tb)

        if lbe.overlap_AABBs(amin, amax, bmin, bmax):
            points = lbe.tri_tri_intersection(ta, tb)
            pba = lbe.tri_tri_intersection(tb, ta)
            for p in pba:
                tmp = p.uvs[0]
                p.uvs[0] = p.uvs[1]
                p.uvs[1] = tmp
            points.extend(pba)
            for p in points:
                p.faces = [fa.index, fb.index]
            if len(points) == 2:
                segments.append(points)
print(len(segments), 'intersection segments')

print('Tracing intersections...')
mat = bpy.data.materials.new("mat_intersection")
mat.diffuse_color = [1,1,0]
mat.diffuse_intensity = 1
mat.emit = 1
mat.use_shadeless = True
                
for i, s in enumerate(segments):
    co = [p.xyz for p in s]
    so = lbe.pydata_to_polyline(co,
                                name="intersection_"+str(i),
                                thickness=1e-2)
    so.data.materials.append(mat)
    so.hide_render = True


for o in obj:
    o.select = False
    
for imesh, o in enumerate(obj):
    o.select = True
    # export obj
    bpy.ops.export_scene.obj(filepath=pth+'intersected_mesh_'+str(imesh)+'.obj',
                             use_selection=True,
                             use_animation=False,
                             use_mesh_modifiers=True,
                             use_edges=True,
                             use_smooth_groups=False,
                             use_smooth_groups_bitflags=False,
                             use_normals=False,
                             use_uvs=True,
                             use_materials=True,
                             use_triangles=False,
                             use_nurbs=False,
                             use_vertex_groups=False,
                             use_blen_objects=True,
                             group_by_object=False,
                             group_by_material=True,
                             keep_vertex_order=True,
                             axis_forward='-Z', axis_up='Y', global_scale=1.0, path_mode='AUTO')
    o.select = False

    """
    f = open(pth+'intersection_segments_'+str(imesh)+'.dat', 'w')
    for s in segments:
        for p in s:
            for x in [p.faces[imesh], p.uvs[imesh].x, p.uvs[imesh].y]:
                f.write(str(x) + ' ')
            f.write('\n')
    f.close()
    """
    uvdata = o.data.uv_layers.active.data
    uv = numpy.zeros((len(segments),4))
    for ise, s in enumerate(segments):
        for ipt, p in enumerate(s):
            iface = p.faces[imesh]
            face = o.data.polygons[iface]
            iloop = [face.loop_start + jloop for jloop in range(face.loop_total)]
            baryco = [1.0 - p.uvs[imesh].x - p.uvs[imesh].y, p.uvs[imesh].x, p.uvs[imesh].y]
            for jloop in range(3):
                for jdim in range(2):
                    uv[ise,2*ipt + jdim] += baryco[jloop]*uvdata[face.loop_start + jloop].uv[jdim]

    numpy.savetxt(pth+'intersection_segments_uv_'+str(imesh)+'.dat', uv)



##### make textures
print('Making textures...')
os.system('python '+pth+'make_intersection_trace_texture.py')

print('Loading textures...')
for o in scene.objects:
    o.select = False
    
for imesh in range(2):
    o = obj[imesh]
    o.select = True
    mat = bpy.data.materials.new('mat_'+str(imesh))
    mat.use_shadeless = True
    tex = bpy.data.textures.new('tex_'+str(imesh), 'IMAGE')
    img = bpy.data.images.load(filepath=pth+'intersection_trace_'+str(imesh)+'.png')
    tex.image = img
    slot = mat.texture_slots.add()
    slot.texture = tex
    slot.texture_coords = 'UV'
    if imesh == 1:
        mat.diffuse_color = [0.509, 0.886, 0.470]
        slot.blend_type = 'MULTIPLY'
    o.data.materials.append(mat)

print('Ready!')
