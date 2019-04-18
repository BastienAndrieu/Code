import bpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_diffgeom as ldg
import lib_color as lco
import lib_chebyshev as lch

sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl

import math
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

"""
## Set Lighting
lamp = lbu.add_point_light(name="lamp",
                           energy=1.2,
                           shadow_method='RAY_SHADOW',
                           shadow_ray_samples=16,
                           shadow_soft_size=2.0,
                           location=[3.75,1.65,3.20])
for i in range(4):
    lamp.layers[i] = True
"""

## Set Camera
cam = scene.camera
for i in range(4):
    cam.layers[i] = True
cam.location = [2.102,1.798,1.104]
cam.rotation_euler = numpy.radians(numpy.array([66.7,0.778,132.2]))
cam.data.angle = numpy.radians(37.72)
################################






###############################################
nsamp = 200
zsamp = numpy.linspace(-1,1,nsamp+1)
cmap = lco.sample_colormap('myBuRd', nsamp+1)
def div_cmap2(z):
    i = numpy.floor(0.5*(z + 1)*nsamp)
    if i == nsamp:
         return cmap[-1]
    else:
        t = (z - zsamp[i])/(zsamp[i+1] - zsamp[i])
        return cmap[i]*(1 - t) + cmap[i+1]*t
###############################################
def pseudocolors(v, cmap, sym=False):
    n = len(v)
    m = len(cmap)-1
    if sym:
        vmax = numpy.amax(numpy.absolute(v))
        vmin = -vmax
    else:
        vmin = numpy.amin(v)
        vmax = numpy.amax(v)
    w = (v - vmin)/(vmax - vmin)
    c = numpy.zeros((n,3))
    for i in range(n):
        j = int(math.floor(w[i]*m))
        if j == m:
            c[i] = cmap[-1]
        else:
            t = w[i]*float(m-2) - float(j)
            c[i] = cmap[j]*(1 - t) + cmap[j+1]*t
    return c
###############################################






suffixe = '_eos'#''#






nf = 31
for iface in range(nf):
    strf = format(iface+1,'03')
    print('\nface #',iface+1)
    print('   load pydata...')
    
    tri = numpy.loadtxt(pthin + 'brepmesh'+suffixe+'/tri_' + strf + '.dat', dtype=int)-1
    uv = numpy.loadtxt(pthin + 'brepmesh'+suffixe+'/uv_' + strf + '.dat', dtype=float)

    c = myl.readCoeffs(pthin + 'brepmesh'+suffixe+'/c_' + strf + '.cheb')
    c_u, c_v = lch.diff2(c)
    c_uu, c_uv = lch.diff2(c_u)
    c_uv, c_vv = lch.diff2(c_v)
    xyz = chebval2d(uv[:,0], uv[:,1], c).T
    
    verts = [[float(x) for x in p] for p in xyz]
    faces = [[int(v) for v in t] for t in tri]

    print('   pydata to mesh...')
    obj = lbu.pydata_to_mesh(verts,
                             faces,
                             name='face_'+strf)
    scene.objects.active = obj
    lbe.set_smooth(obj)

    print('   diffgeom...')
    # Diffgeom
    nor = []
    curvatures = []
    for u in uv:
        dxyz_du    = chebval2d(u[0], u[1], c_u)
        dxyz_dv    = chebval2d(u[0], u[1], c_v)
        d2xyz_du2  = chebval2d(u[0], u[1], c_uu)
        d2xyz_dudv = chebval2d(u[0], u[1], c_uv)
        d2xyz_dv2  = chebval2d(u[0], u[1], c_vv)
        n, Kmin, Kmax, Kgauss, Kmean = ldg.all_geomdiff(dxyz_du, dxyz_dv, d2xyz_du2, d2xyz_dudv, d2xyz_dv2)
        nor.append(n)
        curvatures.append([Kmin, Kmax, Kgauss, Kmean])

    # Vertex colors
    print('   vertex colors...')
    msh = obj.data
    msh.vertex_colors.new()
    vertexcolor = msh.vertex_colors[0].data
    polys = msh.polygons

    """
    val = numpy.asarray(curvatures)
    pco = pseudocolors(val[:,2], cmap, sym=True)
    """
    
    for f in polys:
        verts = f.vertices
        for k in range(f.loop_total):
            l = f.loop_start + k
            
            # Gauss map (normal)
            for j in range(3):
                vertexcolor[l].color[j] = 0.5*(nor[verts[k]][j] + 1.0)
            
            # Gaussian curvature
            #vertexcolor[l].color = pco[verts[k]]
    # material
    mat = bpy.data.materials.new('mat_face_'+strf)
    mat.use_vertex_color_paint = True
    mat.use_shadeless = True
    obj.data.materials.append(mat)
        
