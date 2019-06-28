import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
from lib_compgeom import quickhull2d, minimum_area_OBB
from lib_linalg import matmul

###############################################
# GENERATE POINT CLOUD
args = sys.argv
if len(args) < 2:  
    dim = 2
    n1 = 10
    n2 = 50
    s = 0.15

    p1 = 2*numpy.random.rand(n1,dim) - 1
    p = numpy.empty((0,dim), float)
    for i in range(n1):
        ni = numpy.random.randint(n2)
        pi = s*(2*numpy.random.rand(ni,dim) - 1)
        for j in range(dim):
            pi[:,j] = pi[:,j] + p1[i,j]
        p = numpy.vstack((p,pi))

else:
    p = numpy.loadtxt('/stck/bandrieu/Bureau/CYPRES/Geometry/computational_geometry/xy%d.dat' %int(args[1]))

###############################################


###############################################
# GET CONVEX HULL
hull = quickhull2d(p)
###############################################

###############################################
# GET MINIMUM-AREA OBB
center, ranges, axes = minimum_area_OBB(p)
###############################################

###############################################
# VISU
fig, ax = plt.subplots()

ax.add_patch(mpatches.Polygon(
    xy=p[hull],
    closed=True,
    fc='y',
    alpha=0.5,
    ec='r'
))
ax.plot(p[hull,0], p[hull,1], 'ro', mec='r', mfc='w')
ax.plot(p[:,0], p[:,1], 'k.')

box = numpy.zeros((2,4))
box[:,0] = -ranges[0]*axes[:,0] - ranges[1]*axes[:,1]
box[:,1] =  ranges[0]*axes[:,0] - ranges[1]*axes[:,1]
box[:,2] =  ranges[0]*axes[:,0] + ranges[1]*axes[:,1]
box[:,3] = -ranges[0]*axes[:,0] + ranges[1]*axes[:,1]
for i in range(4):
    box[:,i] = box[:,i] + center
ax.add_patch(mpatches.Polygon(
    xy=box.T,
    closed=True,
    fc='c',
    alpha=0.2,
    ec='b'
))

cl = ['r','g']
for i in range(2):
    ax.arrow(center[0], center[1], ranges[i]*axes[0,i], ranges[i]*axes[1,i], length_includes_head=True, color=cl[i])

ax.set_aspect('equal')
plt.show()

###############################################
