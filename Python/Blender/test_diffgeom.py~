import bpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_color as lco

sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl

import numpy
from numpy.polynomial.chebyshev import chebval2d

########################################################
pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/BRep/'
################################
## Set Scene
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

lbu.set_scene(resolution_x=800,
              resolution_y=800,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=16,
              use_environment_light=True,
              environment_energy=0.3,
              environment_color='PLAIN')

## Set Lighting
lamp = lbu.add_point_light(name="lamp",
                           energy=1.2,
                           shadow_method='RAY_SHADOW',
                           shadow_ray_samples=16,
                           shadow_soft_size=2.0,
                           location=[3.75,1.65,3.20])
for i in range(4):
    lamp.layers[i] = True

## Set Camera
cam = scene.camera
for i in range(4):
    cam.layers[i] = True
cam.location = [2.102,1.798,1.104]
cam.rotation_euler = numpy.radians(numpy.array([66.7,0.778,132.2]))
cam.data.angle = numpy.radians(37.72)
################################

imgchecker = bpy.data.images.load(filepath='/d/bandrieu/GitHub/These/memoire/figures/code/BRep/checker.png')
texchecker = bpy.data.textures.new('texture_checker', 'IMAGE')
texchecker.image = imgchecker

clf = numpy.loadtxt(pthout + 'face_color.dat')
clf = lco.cc_hsv(clf, fs=1.2, fv=1.0)

nf = 31
for iface in range(nf):
    strf = format(iface+1,'03')
    print('\nface #',iface+1)
    print('   load pydata...')
    if iface < 9:
        tri = numpy.loadtxt(pthin + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
        uv = numpy.loadtxt(pthin + 'brepmesh/uv_' + strf + '.dat', dtype=float)
    else:
        tri = numpy.loadtxt(pthin + 'brepmesh_eos/tri_' + strf + '.dat', dtype=int)-1
        uv = numpy.loadtxt(pthin + 'brepmesh_eos/uv_' + strf + '.dat', dtype=float)

    c = myl.readCoeffs(pthin + 'brepmesh_eos/c_' + strf + '.cheb')
    xyz = chebval2d(uv[:,0], uv[:,1], c).T
    
    verts = [[float(x) for x in p] for p in xyz]
    faces = [[int(v) for v in t] for t in tri]

    print('   pydata to mesh...')
    obj = lbu.pydata_to_mesh(verts,
                             faces,
                             name='face_'+strf)
    scene.objects.active = obj

    lbe.set_smooth(obj)
    
    # UV coordinates
    bpy.ops.object.mode_set(mode='EDIT')
    print('   unwrap UVs...')
    bpy.ops.uv.unwrap(method='ANGLE_BASED',
                      fill_holes=True,
                      correct_aspect=True,
                      use_subsurf_data=False,
                      margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')
    uvlayer = obj.data.uv_layers.active
    print('   edit UVs...')
    for i, f in enumerate(obj.data.polygons):
        for j in range(f.loop_total):
            k = f.loop_start + j
            for l in range(2):
                uvlayer.data[k].uv[l] = 0.5*(uv[f.vertices[j],l] + 1.0)

    # material
    mat = bpy.data.materials.new('mat_face_'+strf)
    mat.diffuse_color = clf[iface]
    mat.diffuse_intensity = 1
    mat.specular_intensity = 0
    mat.specular_hardness = 30
    mat.use_transparency = False
    #
    slot = mat.texture_slots.add()
    slot.texture = texchecker
    slot.texture_coords = 'UV'
    slot.blend_type = 'MULTIPLY'
    slot.diffuse_color_factor = 0.09
    obj.data.materials.append(mat)
    #mat.use_shadeless = True

    # change layer
    obj.layers[3] = True
    if iface < 9:
        obj.layers[1] = True
    else:
        obj.layers[2] = True
    obj.layers[0] = False
        
