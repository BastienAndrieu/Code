import bpy

pth = '/d/bandrieu/GitHub/Code/Python/Blender/'

scene = bpy.context.scene

# clear scene
for ob in scene.objects:
    if ob.type == 'CAMERA': 
        ob.select = False
    else: 
        ob.select = True
        
bpy.ops.object.delete()


for imesh in range(2):
    bpy.ops.import_scene.obj(filepath=pth+'intersected_mesh_'+str(imesh)+'.obj',
                             axis_forward='-Z', axis_up='Y')
"""
obj = []
for ob in scene.objects:
    if ob.type == 'MESH':
        obj.append(ob)

# set camera
for o in scene.objects:
    o.select = True
bpy.ops.view3d.camera_to_view_selected()
for o in scene.objects:
    o.select = False

for imesh, o in enumerate(obj):
    o.select = True
    #bpy.ops.object.material_slot_remove()
    mat = bpy.data.materials.new('mat_'+str(imesh))
    ob.active_material_index = 1
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
"""
