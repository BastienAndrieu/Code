import numpy, math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

######################################
def overlap_intervals(ab, cd):
    a = min(ab[0], ab[1])
    b = max(ab[0], ab[1])
    c = min(cd[0], cd[1])
    d = max(cd[0], cd[1])
    return min(b, d) <= max(a, b)
######################################
def overlap_boxes(ab, cd):
    for i in range(2):
        overlap = overlap_intervals(ab[:,i], cd[:,i])
        if not overlap: return overlap
    return overlap
######################################
def get_bounding_box(p):
    dim = len(p[0])
    xmin = p[0].copy()
    xmax = p[0].copy()
    for i in range(1,len(p)):
        for j in range(dim):
            xmin[j] = min(xmin[j], p[i][j])
            xmax[j] = max(xmax[j], p[i][j])
    return xmin, xmax
######################################


nx = 4
ny = nx

x = numpy.linspace(0,1,nx+1)
y = numpy.linspace(0,1,ny+1)

dx = x[1] - x[0]
dy = y[1] - y[0]

z = numpy.zeros((nx,ny), dtype=bool)

# read polyline
#t = numpy.linspace(0,2.0*numpy.pi,100)
#p = 0.5 + 0.33*numpy.vstack([numpy.cos(t),numpy.sin(t)]).T
p = 0.5*(1.0 + numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/data/BRep/faces/contour_int_002_1.dat'))
np = len(p)

# rasterize
pixel = numpy.zeros((2,2))
for k in range(np-1):
    q = p[k:k+2]
    qmin, qmax = get_bounding_box(q)
    box = numpy.vstack([qmin, qmax])
    imin = int(math.floor(qmin[0]/dx))
    imax = int(math.ceil(qmax[0]/dx))
    jmin = int(math.floor(qmin[1]/dy))
    jmax = int(math.ceil(qmax[1]/dy))
    for i in range(max(0,imin), min(nx,imax)):
        #pixel[0,0] = x[i]
        #pixel[1,0] = x[i+1]
        for j in range(max(0,jmin), min(ny,jmax)):
            #pixel[0,1] = y[j]
            #pixel[1,1] = y[j+1]
            z[i,j] = True
            #if not z[i,j]:
                #z[i,j] = overlap_boxes(box, pixel)
                #print z[i,j]

# plot
fig, ax = plt.subplots()
for i in range(nx):
    for j in range(ny):
        if z[i,j]:
            r = patches.Rectangle((i*dx,j*dy), dx, dy,
                                  facecolor='r',
                                  linewidth=0)
            ax.add_patch(r)
for i in range(nx+1):
    ax.plot(x[[i,i]], [0.0,1.0], 'b-')
for j in range(ny+1):
    ax.plot([0.0,1.0], y[[j,j]], 'b-')
ax.plot(p[:,0], p[:,1], 'k.-')
ax.set_aspect('equal')
plt.show()
