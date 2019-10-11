import numpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_halfedge as lhe

pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
iface = 5
strf = format(iface,'03')

print 'read data...'
tri = numpy.loadtxt(pth + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pth + 'brepmesh/xyz_' + strf + '.dat', dtype=float)
print '   ok.'

print 'make halfedge DS...'
mesh = lhe.pydata_to_SurfaceMesh(xyz[:,0:2], tri)
print '   ok.'

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
print 'plot...'
lhe.plot_mesh(mesh,
              ax,
              faces=False,
              edges=False,
              halfedges=True,
              vertices=False,
              boundaries=True,
              v2h=False,
              v2f=False,
              count_from_1=False)
print '   ok.'
ax.set_aspect('equal')
plt.show()
