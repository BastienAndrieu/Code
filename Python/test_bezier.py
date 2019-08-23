import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
#b = b[0:4]
"""
b = numpy.array([(0,0), (2,1)])
"""
b = b + 0.4*(2*numpy.random.rand(b.shape[0], b.shape[1]) - 1)


db = lbez.diff(b)

n = 100
t = numpy.linspace(0,1,n)
"""
p = numpy.zeros((n,2))
for i in range(n):
    f, bl, br = lbez.de_casteljau(b, t[i])
    for j in range(2):
        p[i,j] = f[j]
"""

p = lbez.eval_bezier_curve(b, t)
dp = lbez.eval_bezier_curve(db, t)
#print dp

fig, ax = plt.subplots()

aabb = lbez.AABB_2d_bezier_curve(b)
ctr, rng, axe = lbez.OBB_2d_bezier_curve(b)

ax.plot(p[:,0], p[:,1], 'k-')
ax.plot(b[:,0], b[:,1], 'r.-')

#ax.quiver(p[:,0], p[:,1], dp[:,0], dp[:,1], color='b')
#ax.quiver(p[:,0], p[:,1], -dp[:,1], dp[:,0], color='b')

#ax.plot(dp[:,0], dp[:,1], 'b')

ax.add_patch(
    patches.Rectangle(
        aabb[0],
        aabb[1][0] - aabb[0][0],
        aabb[1][1] - aabb[0][1],
        ec='g',
        fill=False
    )
)

ax.add_patch(
    patches.Rectangle(
        ctr - rng[0]*axe[:,0] - rng[1]*axe[:,1],
        2*rng[0],
        2*rng[1],
        ec='b',
        angle=numpy.degrees(numpy.arctan2(axe[1][0], axe[0][0])),
        fill=False
    )
)

ax.set_aspect('equal')
plt.show()

















fig, ax = plt.subplots()

br = lbez.reparameterize_bezier_curve(b, start=0.3, end=0.7)
p = lbez.eval_bezier_curve(b, t)
pr = lbez.eval_bezier_curve(br, t)

ax.plot(p[:,0], p[:,1], 'k:')
ax.plot(pr[:,0], pr[:,1], 'k-')
ax.plot(br[:,0], br[:,1], 'r.-')

aabb = lbez.AABB_2d_bezier_curve(br)
ctr, rng, axe = lbez.OBB_2d_bezier_curve(br)

ax.add_patch(
    patches.Rectangle(
        aabb[0],
        aabb[1][0] - aabb[0][0],
        aabb[1][1] - aabb[0][1],
        ec='g',
        fill=False
    )
)

ax.add_patch(
    patches.Rectangle(
        ctr - rng[0]*axe[:,0] - rng[1]*axe[:,1],
        2*rng[0],
        2*rng[1],
        ec='b',
        angle=numpy.degrees(numpy.arctan2(axe[1][0], axe[0][0])),
        fill=False
    )
)

ax.set_aspect('equal')
plt.show()
