import numpy
import matplotlib.pyplot as plt

def cross2d(u, v):
    return u[0]*v[1] - u[1]*v[0]

def squared_norm(u):
    return numpy.sum(u**2)

## random polyline
n = 20
closed = True
t = 2*numpy.pi*numpy.linspace(0,1,n+1)
#r = 1 + 8e-2*(2*numpy.random.rand(n) - 1)
r = 1 + 0.4*numpy.cos(10*numpy.pi*numpy.linspace(0,1,n))
p = numpy.vstack([r*numpy.cos(t[0:n]), r*numpy.sin(t[0:n])]).T


## convexification
nsteps = 20

if closed:
    prange = range(0,n)
else:
    prange = range(1,n-1)

p0 = p.copy()
## plot
fig, ax = plt.subplots()
ax.plot(p0[:,0], p0[:,1], 'b.-')


for step in range(nsteps):
    ptmp = p.copy()
    count = 0
    for i in prange:
        vm1 = p[i] - p[(i+n-1)%n]
        vp1 = p[(i+1)%n] - p[i]
        if cross2d(vm1, vp1) < 0:
            e = vp1 + vm1
            l = numpy.dot(vm1, e)/squared_norm(e)
            ptmp[i] = p[(i+n-1)%n] + l*e
            count += 1
    if count < 1:
        print "no more change after %d step(s)" %step
    else:
        print "%d change(s)" %count
    p = ptmp.copy()
    #ax.plot(p[:,0], p[:,1], 'r.-')
ax.plot(p[:,0], p[:,1], 'r.-')



## plot
"""
fig, ax = plt.subplots()
ax.plot(p0[:,0], p0[:,1], 'b.-')
ax.plot(p[:,0], p[:,1], 'r.-')
"""

ax.set_aspect('equal')
plt.show()

