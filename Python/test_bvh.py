# -*-coding:Latin-1 -*
ROOT = '/d/bandrieu/'#'/home/bastien/'#

import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

import sys
sys.path.append(ROOT+'GitHub/Code/Python')
import lib_morton

cmap = plt.get_cmap('rainbow')

#############################################################
class Primitive:
    def __init__(self, verts=None):
        if verts is None:
            self.verts = []
        else:
            self.verts = verts
            coords_min = numpy.amin(verts, axis=0)
            coords_max = numpy.amax(verts, axis=0)
            dim = len(verts[0])
            self.extents = numpy.zeros(2*dim)
            for i in range(dim):
                self.extents[2*i]   = coords_min[i]
                self.extents[2*i+1] = coords_max[i]
        return
    #
    def get_centroid(self):
        n_verts = len(self.verts)
        if n_verts == 0:
            return None
        elif n_verts == 1:
            return self.verts[0][:]
        else:
            return numpy.sum(self.verts, axis=0)/n_verts
    #
#############################################################
#http://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies.html#fragment-CreateLBVHtreeletsatbottomofBVH-0
class LBVHTreelet:
    def __init__(self, startIndex, nPrimitives):
        self.startIndex = startIndex
        self.nPrimitives = nPrimitives
        return

#############################################################
def random_primitive(dim=2, n_verts_min=1, n_verts_max=5, scale=1, rand_scale=1):
    ctr = numpy.random.rand(dim)
    n_verts = numpy.random.randint(n_verts_min, n_verts_max+1)
    scale *= (1 + 0.5*rand_scale*(2*numpy.random.rand() - 1))
    #
    if dim == 1:
        verts = ctr
    elif dim == 2:
        t = numpy.linspace(-1,1,n_verts+1) + (2*numpy.random.rand(n_verts+1) - 1)/n_verts
        t = numpy.pi*t
        verts = numpy.zeros((n_verts,dim))
        for i in range(n_verts):
            r = scale*(1 + 0.5*(2*numpy.random.rand() - 1)/n_verts)
            verts[i][0] = ctr[0] + r*numpy.cos(t[i])
            verts[i][1] = ctr[1] + r*numpy.sin(t[i])
    else:
        verts = numpy.tile(ctr, (n_verts,1)) + scale*0.5*(2*numpy.random.rand(n_verts,dim) - 1)
    #
    return Primitive(verts)
#############################################################
if False:
    n_primitives = 200
    n_verts_min = 2
    n_verts_max = 6

    dim = 2

    scale = 2.0/n_primitives
    rand_scale = 0.8
    primitives = [
        random_primitive(
            dim,
            n_verts_min,
            n_verts_max,
            scale,
            rand_scale,
        )
        for i in range(n_primitives)
    ]
else:
    pth = ROOT+'Téléchargements/ne_50m_admin/'
    land = 'australia'#'bolivia_mali_iceland'
    xy = numpy.loadtxt(pth+land+'_xy.dat')
    f2v = numpy.loadtxt(pth+land+'_tri.dat', dtype=int)
    primitives = [
        Primitive(verts=xy[f]) for f in f2v
    ]
    dim = xy.shape[1]
    n_primitives = len(primitives)




# compute scene extents 
scene_extents = primitives[0].extents.copy()
for p in primitives[1:]:
    extents = p.extents
    for i in range(dim):
        scene_extents[2*i]   = min(scene_extents[2*i],   extents[2*i])
        scene_extents[2*i+1] = max(scene_extents[2*i+1], extents[2*i+1])

# compute morton codes for primitive centroids
n_bits = 0
while 2**n_bits < n_primitives:
    n_bits += 1

n_bits = 10
n = 1 << n_bits#2**n_bits
invh = [n/(scene_extents[2*i+1] - scene_extents[2*i]) for i in range(dim)]

morton_codes = [0 for p in primitives]
for i, p in enumerate(primitives):
    ctr = p.get_centroid()
    ctr = [int((ctr[j] - scene_extents[2*j])*invh[j]) for j in range(dim)]
    if dim == 2:
        morton_codes[i] = lib_morton.interleave2(ctr[0], ctr[1])
    elif dim == 3:
        morton_codes[i] = lib_morton.interleave3(ctr[0], ctr[1], ctr[2])
    else:
        print 'dim must be 2 or 3'
        exit()

# sort by ascending morton code
order = numpy.argsort(morton_codes)

primitives = [primitives[i] for i in order]
morton_codes = [morton_codes[i] for i in order]


# build treelets
treeletsToBuild = []
start = 0
end = 1
while end < n_primitives:
    mask = 0b00111111111111000000000000000000
    if end == n_primitives-1 or morton_codes[start] & mask != morton_codes[end] & mask:
        # Add entry to treeletsToBuild for this treelet
        nPrimitives = end - start
        if end == n_primitives-1: nPrimitives += 1
        maxBVHNodes = 2*nPrimitives - 1
        treeletsToBuild.append(
            LBVHTreelet(start, nPrimitives)
        )
        start = end
    #
    end += 1

for treelet in treeletsToBuild:
    print treelet.startIndex, treelet.startIndex+treelet.nPrimitives-1





# visualize
if dim == 2:
    fig, ax = plt.subplots()

    patches = [Polygon(p.verts) for p in primitives]
    collect = PatchCollection(patches, cmap=cmap, alpha=0.5)
    collect.set_array(numpy.arange(n_primitives))
    ax.add_collection(collect)

    pboxes = [
        Rectangle(
            xy=p.extents[[0,2]],
            width=p.extents[1] - p.extents[0],
            height=p.extents[3] - p.extents[2],
            fill=False
        )
        for p in primitives
    ]
    pbcollect = PatchCollection(pboxes, cmap=cmap, alpha=0.1)
    pbcollect.set_array(numpy.arange(n_primitives))
    ax.add_collection(pbcollect)
    

    tboxes = []
    for treelet in treeletsToBuild:
        extents = primitives[treelet.startIndex].extents[:]
        x = []
        y = []
        for j in range(treelet.nPrimitives):
            p = primitives[treelet.startIndex+j]
            xj, yj = p.get_centroid()
            x.append(xj)
            y.append(yj)
            for i in range(dim):
                extents[2*i]   = min(extents[2*i],   p.extents[2*i])
                extents[2*i+1] = max(extents[2*i+1], p.extents[2*i+1])
        tboxes.append(
            Rectangle(
                xy=extents[[0,2]],
                width=extents[1] - extents[0],
                height=extents[3] - extents[2],
                fill=False
            )
        )
        #ax.plot(x, y, '.-')
    tbcollect = PatchCollection(tboxes, alpha=0.25)
    tbcollect.set_array(numpy.arange(len(treeletsToBuild)))
    ax.add_collection(tbcollect)

    ax.set_xlim(scene_extents[0], scene_extents[1])
    ax.set_ylim(scene_extents[2], scene_extents[3])
    ax.set_aspect('equal')
    plt.show()
