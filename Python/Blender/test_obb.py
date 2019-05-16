import bpy
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_chebyshev as cheb
import lib_blender_util as lbu
import numpy
from numpy.polynomial.chebyshev import chebval, chebgrid2d

sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl

##################################################################
scene = bpy.context.scene
cam = scene.camera
cam.hide = True
lbu.set_scene(resolution_x=800,
              resolution_y=600,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=10,
              use_environment_light=True,
              environment_energy=0.2,
              environment_color='PLAIN')

# clear Blender scene
lbu.clear_scene(meshes=True, lamps=False, cameras=False)

# read polynomial coefficients in Chebyshev basis
#filecoefs = '/stck/bandrieu/Bureau/Manuscrit/memoire_bak_29_03_2019/figures/code/parametric_patch.cheb'
filecoefs = '/d/bandrieu/GitHub/FFTsurf/test/coeffstest/C1_test51.txt'
c = cheb.read_polynomial2(filecoefs)

# compute Oriented Bounding Box
center, ranges, axes = cheb.obb_chebyshev2(c)

# add tensor product patch
m = 100
u = numpy.linspace(-1,1,m)
xyz = chebgrid2d(u, u, c)
myl.addTensorProductPatch(xyz[0], xyz[1], xyz[2],
                          name="patch",
                          periodu=False, periodv=False,
                          location=[0,0,0],
                          smooth=True,
                          color=[1,1,1], alpha=1, emit=0.0)

# add OBB as polyhedron
obb = lbu.obb_to_mesh(center, ranges, axes)
obb.name = 'OBB0'
matobb = bpy.data.materials.new('mat_OBB')
matobb.diffuse_color = [.8, 0.5, 0.25]
matobb.diffuse_intensity = 1
matobb.specular_intensity = 0
matobb.use_transparency = True
matobb.alpha = 0
#matobb.emit = 0.5
#matobb.raytrace_transparency.fresnel = 2
matobb.use_raytrace = False
obb.data.materials.append(matobb)


obb.show_wire = True
obb.show_all_edges = True
obb.show_transparent = True



"""
# add OBB axes
for i in range(3):
    axi = lbu.pydata_to_polyline([center, center+ranges[i]*axes[:,i]],
                                 name='axe_'+str(i),
                                 thickness=1e-2)
    axi.hide_render = True                     
"""


#####
for ivar in range(2):
    for ival in range(2):
        a = cheb.bivariate_to_univariate(c, ivar, ival)
        xyz = chebval(u, a)
        obj = lbu.pydata_to_polyline(xyz.T,
                                     name='iso_'+str(ival)+'_'+str(ivar),
                                     thickness=2e-3)
        obj.hide_render = True
        center, ranges, axes = cheb.obb_chebyshev1(a)
        obb = lbu.obb_to_mesh(center, ranges, axes)
        obb.name = 'OBB'+str(1 + ival + 2*ivar)
        obb.data.materials.append(matobb)
        obb.show_wire = True
        obb.show_all_edges = True
        obb.show_transparent = True
        obb.select = True

# freestyle
bpy.ops.object.select_all(action='DESELECT')

for i in range(5):
    obj = bpy.data.objects['OBB'+str(i)]
    obj.show_wire = True
    obj.show_all_edges = True
    obj.show_transparent = True
    obj.select = True
bpy.ops.view3d.camera_to_view_selected()
cam.data.sensor_width *= 1.01
bpy.ops.group.create(name='obb_group')

scene.render.use_freestyle = True
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True
bpy.ops.scene.freestyle_lineset_add()

for lineset in freestyle.linesets:
    lineset.select_silhouette = False
    lineset.select_border = False
    lineset.select_crease = True
    lineset.select_by_group = True
    lineset.group = bpy.data.groups['obb_group']
freestyle.linesets[0].visibility = 'VISIBLE'
freestyle.linesets[1].visibility = 'RANGE'
freestyle.linesets[1].qi_start = 1
freestyle.linesets[1].qi_end = 1

for linestyle in bpy.data.linestyles:
    linestyle.caps = "ROUND"
    linestyle.color = [0,0,0]
    linestyle.thickness = 2
    linestyle.geometry_modifiers['Sampling'].sampling = 0.1

bpy.data.linestyles[1].use_dashed_line = True
bpy.data.linestyles[1].dash1 = 7
bpy.data.linestyles[1].gap1 = 7
