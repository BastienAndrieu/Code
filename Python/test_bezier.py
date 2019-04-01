import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_bezier as lbez

b = numpy.array([[0,0],
                 [0.3,1],
                 [1,1],
                 [1,0],
                 [1,-1],
                 [1.5,-0.5],
                 [2,0]])
b = b[0:4]

n = 100
t = numpy.linspace(0,1,n)
p = numpy.zeros((n,2))
for i in range(n):
    f, bl, br = lbez.de_casteljau(b, t[i])
    for j in range(2):
        p[i,j] = f[j]

fig, ax = plt.subplots()

ax.plot(p[:,0], p[:,1], 'k-')
ax.plot(b[:,0], b[:,1], 'r.-')

ax.set_aspect('equal')
plt.show()
