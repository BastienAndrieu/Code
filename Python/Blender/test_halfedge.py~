import numpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_halfedge as lhe

pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
iface = 2
strf = format(iface,'03')

tri = numpy.loadtxt(pth + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pth + 'brepmesh/xyz_' + strf + '.dat', dtype=float)

mesh = lhe.pydata_to_SurfaceMesh(xyz, tri)


