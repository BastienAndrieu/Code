import numpy
import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
from lib_trees import TreeNode

def plot_tree(node, ax):
    if node.is_leaf():
        x = [node.extents[0], node.extents[1], node.extents[1], node.extents[0], node.extents[0]]
        y = [node.extents[2], node.extents[2], node.extents[3], node.extents[3], node.extents[2]]
        ax.plot(x, y, 'k-')
    else:
        for child in node.children:
            plot_tree(child, ax)
    return 

def place_points_in_tree(node, points, max_points_per_node=1):
    if len(node.ipoints) > max_points_per_node:
        node.split(points)
        for child in node.children:
            place_points_in_tree(child, points, max_points_per_node)
    return



npts = 100
#xy = numpy.random.rand(npts,2)
t = 2*numpy.pi*(numpy.linspace(0,1,npts))
xy = numpy.vstack([numpy.cos(t), numpy.sin(t)]).T
xy = xy + 0.1*(2*numpy.random.rand(npts,2) - 1)

xymin = numpy.amin(xy, axis=0)
xymax = numpy.amax(xy, axis=0)


root = TreeNode(
    extents=(xymin[0], xymax[0], xymin[1], xymax[1]),
    ipoints=[i for i in range(npts)]
)

"""
root.split(points = xy)
for ichild, child in enumerate(root.children):
    print 'child #%d contains %d points\n' % (ichild, len(child.ipoints))
    print child.is_leaf()
"""

place_points_in_tree(root, xy, max_points_per_node=1)

# find closest point in quadtree
p = 2*numpy.random.rand(2) - 1



fig, ax = plt.subplots()
plot_tree(root, ax)

ax.plot(xy[:,0], xy[:,1], 'r.')

ax.set_aspect('equal')
plt.show()
