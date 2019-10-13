ROOT = '/home/bastien/'#'/d/bandrieu/'#

import bpy
import numpy
from numpy.polynomial.chebyshev import chebval, chebgrid2d, chebval2d

import sys
sys.path.append(ROOT + 'GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_cadcheb as lcad
import lib_chebyshev as lch


lbu.clear_scene(True, True, True)

cs = lch.read_polynomial2(ROOT+'GitHub/FFTsurf/test/coeffstest/C2_test10.txt')
m = 100
u = numpy.linspace(-1, 1, m)
xyz = chebgrid2d(u, u, cs)
v, f = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
surf = lbu.pydata_to_mesh(
    v,
    f,
    name='surface'
)
lbe.set_smooth(surf)

hmin = 1e-3
hmax = 1
tolchord = 1e-3

N = 6
cc = (2*numpy.random.rand(N,2) - 1)/numpy.tile(numpy.arange(1,N+1)**2, (2,1)).T
cc[0,:] = 0
cc = 3*cc

xyz, uv, t = lcad.discretize_curve_on_surface(
    cc,
    cs,
    hmin,
    hmax,
    tolchord,
    n0=20
)

curv3 = lbu.pydata_to_mesh(
    xyz.T,
    faces=[],
    edges=[(i, i+1) for i in range(len(t)-1)],
    name='curve3'
)

uvw = numpy.vstack([uv, numpy.zeros(len(t))])
curv2 = lbu.pydata_to_mesh(
    uvw.T,
    faces=[],
    edges=[(i, i+1) for i in range(len(t)-1)],
    name='curve2'
)
curv2.layers[1] = True
curv2.layers[0] = False



w = numpy.linspace(-1,1,300)
uv = chebval(w, cc)
xyz = chebval2d(uv[0], uv[1], cs)
unif3 = lbu.pydata_to_mesh(
    xyz.T,
    faces=[],
    edges=[(i, i+1) for i in range(len(w)-1)],
    name='uniform3'
)


