ROOT = '/home/bastien/'#'/d/bandrieu/'#

import bpy
import numpy
from numpy.polynomial.chebyshev import chebval

import sys
sys.path.append(ROOT + 'GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_cadcheb as lcad


lbu.clear_scene(True, True, True)

hmin = 1e-3
hmax = 1
tolchord = 1e-3

N = 8
c = (2*numpy.random.rand(N,3) - 1)/numpy.tile(numpy.arange(1,N+1)**2, (3,1)).T
c[0,:] = 0
c = 10*c

xyz, t = lcad.discretize_curve(
    c,
    hmin,
    hmax,
    tolchord,
    n0=20
)

adap = lbu.pydata_to_mesh(
    xyz.T,
    faces=[],
    edges=[(i, i+1) for i in range(len(t)-1)],
    name='adaptive'
)

u = numpy.linspace(-1,1,200)
unif = lbu.pydata_to_mesh(
    chebval(u, c).T,
    faces=[],
    edges=[(i, i+1) for i in range(len(u)-1)],
    name='uniform'
)


