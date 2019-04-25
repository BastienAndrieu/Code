import bpy
import numpy
from numpy.polynomial.chebyshev import chebval
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_chebyshev as cheb
import lib_EoS as eos

sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl

###############################
def fr(x):
    return 0.2*(1.0 + 1.2*x[0] + 0.2*x[1] - 0.6*x[2])
###############################

# spine's Bezier control points
b = numpy.array([
    [0., -0.2570,  0.0],
    [1.,  0.3086, -0.6],
    [2.,  0.3045, -0.6],
    [3., -0.1435,  0.4],
    [4.,  0.0814,  0.6]])

# convert to Chebyshev polynomial
N = len(b)-1
B2C = cheb.B2Cmatrix(N+1)
pg = numpy.zeros(b.shape)
for j in range(3):
    for i in range(N+1):
        for k in range(N+1):
            pg[i,j] += B2C[i,k]*b[k,j]
# 1st and 2nd polynomial derivatives
dpg = cheb.diff(pg)
d2pg = cheb.diff(dpg)

# sampling
m = 200
n = m
m1 = 20
m2 = 120
dm = 12
u = numpy.linspace(-1,1,m)
v = numpy.pi*numpy.linspace(-1,1,n+1)

# spine curve
g = chebval(u, pg)
dg = chebval(u, dpg)
d2g = chebval(u, d2pg)

gc = chebval(cheb.cgl_nodes(N), pg)
pr = cheb.fcht(fr(gc))
dpr = cheb.diff(pr)
r = chebval(u, pr)
dr = chebval(u, dpr)

# envelope
e1 = eos.one_parameter_EoS(g, dg, d2g, r, dr, v)
e2 = eos.canal_surface(g, dg, d2g, r, dr, v)

############################################################
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

lbu.set_scene(resolution_x=1400,
              resolution_y=960,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=2,
              use_environment_light=False)

lbu.add_point_light(name="lamp",
                    energy=1,
                    shadow_method='NOSHADOW',
                    location=[-3.83856, -4.3118, 6.04704])

scene = bpy.context.scene
cam = scene.camera
cam.location = [14.6387, -19.52351, 0.52857]
cam.rotation_euler = numpy.array([87.975, 0.62, 393.916])*numpy.pi/180.0
cam.data.lens_unit = "FOV"
cam.data.angle = 6.1*numpy.pi/180.0
cam.data.shift_x = 0.018
cam.data.shift_y = 0.03


eos1 = myl.addTensorProductPatch(e1[:,:,0], e1[:,:,1], e1[:,:,2],
                                 name="envelope_1",
                                 periodu=False, periodv=True,
                                 location=[0,0,0],
                                 smooth=True,
                                 color=[0.527, 0.800, 0.213], alpha=0.68, emit=1.0)
eos1.layers[1] = True
eos1.layers[0] = False

eos2 = myl.addTensorProductPatch(e2[:,:,0], e2[:,:,1], e2[:,:,2],
                                 name="envelope_2",
                                 periodu=False, periodv=True,
                                 location=[0,0,0],
                                 smooth=True,
                                 color=[0.527, 0.800, 0.213], alpha=0.68, emit=1.0)
