# -*-coding:Latin-1 -*
ROOT = '/d/bandrieu/'#'/home/bastien/'#

import numpy

import sys
sys.path.append(ROOT+'GitHub/Code/Python')
import lib_halfedge as lhe

print 'read data...'
if False:
    pth = ROOT+'GitHub/FFTsurf/test/demo_EoS_brep/'
    iface = 5
    strf = format(iface,'03')
    tri = numpy.loadtxt(pth + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
    xyz = numpy.loadtxt(pth + 'brepmesh/xyz_' + strf + '.dat', dtype=float)
else:
    pth = ROOT+'Téléchargements/ne_50m_admin/'
    land = 'bolivia'#'bolivia_mali_iceland'
    xyz = numpy.loadtxt(pth+land+'_xy.dat')
    tri = numpy.loadtxt(pth+land+'_tri.dat', dtype=int)
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
              boundaries=False,
              v2h=False,
              v2f=False,
              count_from_1=False)
print '   ok.'
ax.set_aspect('equal')
plt.show()
