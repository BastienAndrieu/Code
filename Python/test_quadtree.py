import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
from lib_trees import TreeNode, place_points_in_tree, get_max_depth, flatten_tree

def plot_tree(node, ax, show_depth=False, color='0.8', linestyle='-', linewidth=1):
    if node.is_leaf():
        x = [node.extents[0], node.extents[1], node.extents[1], node.extents[0], node.extents[0]]
        y = [node.extents[2], node.extents[2], node.extents[3], node.extents[3], node.extents[2]]
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
        if show_depth:
            x, y = [0.5*(node.extents[2*i] + node.extents[2*i+1]) for i in range(2)]
            ax.text(x, y, str(node.depth), color='r')
    else:
        for child in node.children:
            plot_tree(child, ax, show_depth, color, linestyle, linewidth)
    return 

"""def place_points_in_tree(node, points, max_points_per_node=1):
    if len(node.ipoints) > max_points_per_node:
        node.split(points)
        for child in node.children:
            place_points_in_tree(child, points, max_points_per_node)
    return"""

def plot_full_quadtree(depth, extents, ax, color='b', linestyle=':'):
    n = 2**depth + 1
    x = numpy.linspace(extents[0], extents[1], n)
    y = numpy.linspace(extents[2], extents[3], n)
    for i in range(n):
        ax.plot(x[i]*numpy.ones(n), y, color=color, linestyle=linestyle)
        ax.plot(x, y[i]*numpy.ones(n), color=color, linestyle=linestyle)
    return


"""
npts = 1000
#xy = numpy.random.rand(npts,2)
t = 2*numpy.pi*(numpy.linspace(0,1,npts))
xy = numpy.vstack([numpy.cos(t), numpy.sin(t)]).T
xy = xy + 0.8*(2*numpy.random.rand(npts,2) - 1)
"""
nclusters = 10
nmax_per_cluster = 100
noise = 0.3
xyc = 2*numpy.random.rand(nclusters,2) - 1
xy = numpy.empty((0,2), float)
for c in xyc:
    ni = numpy.random.randint(nmax_per_cluster)
    xy = numpy.vstack([
        xy,
        numpy.tile(c, (ni,1)) + noise*(2*numpy.random.rand(ni,2) - 1)
    ])
npts = len(xy)

xymin = numpy.amin(xy, axis=0)
xymax = numpy.amax(xy, axis=0)


root = TreeNode(
    extents=(xymin[0], xymax[0], xymin[1], xymax[1]),
    ipoints=[i for i in range(npts)]
)


print 'place points in tree...'
place_points_in_tree(root, xy, max_points_per_leaf=1)
print '...ok\n'

"""# find closest point in quadtree
p = 2*numpy.random.rand(2) - 1"""


fig, ax = plt.subplots()
print 'plot tree...'
plot_tree(root, ax)
print '...ok\n'
ax.plot(xy[:,0], xy[:,1], 'r.')

ax.set_aspect('equal')
plt.show()







###########################
print 'get tree depth...'
tree_depth = get_max_depth(root, root.depth)
print 'tree depth = %d' % tree_depth

print 'flatten tree leaves...'
flat_tree = flatten_tree(root, {}, tree_depth, root.extents)
print '...ok\n'

print 'sort leaves by Morton code...'
m_sorted = sorted(flat_tree.keys())

cmap = plt.get_cmap('rainbow')
nm = len(m_sorted)
tm = numpy.linspace(0,1,nm)
colors = [cmap(tm[i]) for i in range(nm)]

print '...ok\n'



PLOT_LEAF_ORDER = True
PLOT_POINT_ORDER = False


fig, ax = plt.subplots()
plot_tree(root, ax)#, color='0.2')#, True)

#plot_full_quadtree(tree_depth, root.extents, ax)

#for m, node in flat_tree.iteritems():
#    x, y = [0.5*(node.extents[2*i] + node.extents[2*i+1]) for i in range(2)]
#    ax.text(x, y, str(m))



print 'plot...'
if PLOT_LEAF_ORDER:
    x = []; y = []
    for j, m in enumerate(m_sorted):
        node = flat_tree[m]
        xm, ym = [0.5*(node.extents[2*i] + node.extents[2*i+1]) for i in range(2)]
        ax.plot(xm, ym, 'o', color=colors[j], zorder=10)
        x.append(xm)
        y.append(ym)
    ax.plot(x, y, '-', color='k', lw=0.5, zorder=5)

if PLOT_POINT_ORDER:
    u = []; v = []
    for j, m in enumerate(m_sorted):
        node = flat_tree[m]
        ax.plot(xy[node.ipoints,0], xy[node.ipoints,1], '.', color=colors[j])
        """for i in node.ipoints:
            u.append(xy[i,0])
            v.append(xy[i,1])
    ax.plot(u, v, 'r.:')"""
print '...ok'

ax.set_aspect('equal')
plt.show()
