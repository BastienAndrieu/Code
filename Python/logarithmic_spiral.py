import numpy
import matplotlib.pyplot as plt


#p = 2*numpy.random.rand(2,2) - 1
p = numpy.array([[0,0],[4,1.5]])

t = numpy.arctan2(p[:,1], p[:,0])
r = numpy.hypot(p[:,0], p[:,1])
lr = numpy.log(r)

det = t[1] - t[0]
la = (lr[0]*t[1] - lr[1]*t[0])/det
a = numpy.exp(la)
b = (lr[1] - lr[0])/det


t = 2*numpy.pi*numpy.linspace(-2,1,200)
r = a*numpy.exp(b*t)
x = r*numpy.cos(t)
y = r*numpy.sin(t)


fig, ax = plt.subplots()
ax.plot(x, y, 'b-')
ax.plot(p[:,0], p[:,1], 'r*')
ax.plot(0,0,'k*')
ax.set_aspect('equal')
ax.set_xlim([-2,4])
ax.set_ylim([-1.5,3])

plt.show()
