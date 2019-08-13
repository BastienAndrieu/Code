import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
from lib_compgeom import quickhull2d, minimal_OBB
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
center, ranges, axes = minimal_OBB(p, 'area')
###############################################

###############################################
# GET MINIMUM-WIDTH OBB
centerW, rangesW, axesW = minimal_OBB(p, 'width')
###############################################
print(ranges, ranges[0]*ranges[1])
print(rangesW, rangesW[0]*rangesW[1])


###############################################
# COVARIANCE ELLIPSE
pm = numpy.mean(p, axis=0)
# covariance matrix
C = numpy.cov(p[:,0] - pm[0], p[:,1] - pm[1])
# eigenvalues and eigenvectors sorted in descending order
eigvals, eigvecs = numpy.linalg.eigh(C)
order = eigvals.argsort()[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:,order]
# anti-clockwise angle to rotate our ellipse by 
vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
theta = numpy.arctan2(vy, vx)
# width and height of ellipse to draw
nstd = 2
width, height = 2*nstd*numpy.sqrt(eigvals)
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
"""
cl = ['r','g']
for i in range(2):
    ax.arrow(center[0], center[1], ranges[i]*axes[0,i], ranges[i]*axes[1,i], length_includes_head=True, color=cl[i])
"""

box = numpy.zeros((2,4))
box[:,0] = -rangesW[0]*axesW[:,0] - rangesW[1]*axesW[:,1]
box[:,1] =  rangesW[0]*axesW[:,0] - rangesW[1]*axesW[:,1]
box[:,2] =  rangesW[0]*axesW[:,0] + rangesW[1]*axesW[:,1]
box[:,3] = -rangesW[0]*axesW[:,0] + rangesW[1]*axesW[:,1]
for i in range(4):
    box[:,i] = box[:,i] + centerW
ax.add_patch(mpatches.Polygon(
    xy=box.T,
    closed=True,
    fc='y',
    alpha=0.2,
    ec='g'
))
"""
cl = ['r','g']
for i in range(2):
    ax.arrow(centerW[0], centerW[1], rangesW[i]*axesW[0,i], rangesW[i]*axesW[1,i], length_includes_head=True, color=cl[i])
"""




ax.add_artist(
    mpatches.Ellipse(
        xy=pm,
        width=width,
        height=height,
        angle=numpy.degrees(theta),
        fill=False
    )
)



ax.set_aspect('equal')
plt.show()

###############################################
