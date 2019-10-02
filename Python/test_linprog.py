ROOT = '/d/bandrieu/'

import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append(ROOT+'GitHub/Code/Python/')
from lib_linprog import lp_solve
from lib_linalg import solve_NxN
import lib_compgeom as lcg

pth = ROOT+'GitHub/FFTsurf/test/linearprogramming/'

numtest = sys.argv[1]



# read problem
f = open(pth + 'lpp' + str(numtest) + '.dat', 'r')
f.readline() # dimension
dim = int(f.readline())
f.readline() # linear objective function
c = numpy.array([float(x) for x in f.readline().split()])
f.readline() # linear constraints
n = int(f.readline())
A = numpy.zeros((n,dim+1))
for i in range(n):
    A[i] = [float(x) for x in f.readline().split()]
f.close()

# solve problem
x0, sing = solve_NxN(A[:dim,:dim], -A[:dim,-1])
print 'x0 = ', x0
stat, x = lp_solve(x0, A, c)

print 'stat = %d\n' % stat
if stat == 0: print 'sol =', x



# visualize
v = numpy.empty((0,2))
for i in range(n-1):
    for j in range(i+1,n):
        det = A[i,0]*A[j,1] - A[i,1]*A[j,0]
        if abs(det) > 1.e-15:
            vij = numpy.array( [(-A[i,2]*A[j,1] + A[j,2]*A[i,1])/det , 
                       (-A[i,0]*A[j,2] + A[i,2]*A[j,0])/det ] )
            inside = True
            for k in range(1,n):
                if k != i and k != j:
                    if A[k,0]*vij[0] + A[k,1]*vij[1] + A[k,2] < 0.0:
                        inside = False
                        break
            if inside:
                v = numpy.vstack( (v, vij) )

fig, ax = plt.subplots()

if len(v) > 0:
    hull = lcg.quickhull2d(v)
    xymin, xymax = lcg.get_bounding_box(v[hull])
    xymid = 0.5*(xymax + xymin)
    xyrng = 0.5*(xymax - xymin)
    xyrng = 1.1*max(xyrng)
    xymin = xymin - xyrng
    xymax = xymax + xyrng

    l = numpy.hstack([hull, hull[0]])
    
    #ax.plot(v[l,0], v[l,1], 'k.-')
else:
    xymin = [-10,-10]
    xymax = [10, 10]
if True:
    t = numpy.linspace(0,10,2)
    xy = numpy.zeros((2,len(t)))
    for i in range(n):
        if abs(A[i,0]) > abs(A[i,1]):
            xy[1] = numpy.linspace(xymin[1],xymax[1],2)
            xy[0] = -(A[i,1]*xy[1] + A[i,2])/A[i,0]
        else:
            xy[0] = numpy.linspace(xymin[0],xymax[0],2)
            xy[1] = -(A[i,0]*xy[0] + A[i,2])/A[i,1]
        xm = 0.5*(xy[:,0] + xy[:,1])
        ax.plot(xy[0], xy[1], '--')
        vec = A[i,:2]/numpy.hypot(A[i,0], A[i,1])
        ax.quiver(xm[0], xm[1], vec[0], vec[1])
        

if stat == 0: 
    ax.quiver(x[0], x[1], c[0], c[1], color='r')
    ax.plot(x[0], x[1], 'ro', markersize=6)


ax.set_xlim(xymin[0], xymax[0])
ax.set_ylim(xymin[1], xymax[1])
ax.set_aspect('equal', adjustable='box')
plt.show()




