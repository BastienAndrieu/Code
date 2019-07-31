import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
from lib_compgeom import smallest_enclosing_ball


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
    p = numpy.loadtxt('/stck/bandrieu/Bureau/CYPRES/Geometry/computational_geometry/xy%d.dat' % int(args[1]))
###############################################


###############################################
# COMPUTE SMALLEST ENCLOSING DISK
ctr, rad = smallest_enclosing_ball(p, randomize=False)
rad = numpy.sqrt(rad)
###############################################


###############################################
# VISU
fig, ax = plt.subplots()

ax.plot(p[:,0], p[:,1], 'k.')

ax.plot(ctr[0], ctr[1], 'b*')

ax.add_artist(
    plt.Circle(
        ctr,
        rad,
        ec='r',
        fc='r',
        alpha=0.3,
        fill=True
    )
)

ax.set_aspect('equal')
ax.set_xlim(ctr[0]-rad, ctr[0]+rad)
ax.set_ylim(ctr[1]-rad, ctr[1]+rad)
plt.show()
###############################################

