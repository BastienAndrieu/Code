import numpy





####################################################################
def get_bounding_box(points, xymrg=0.):
    xymin = numpy.amin(points, axis=0)
    xymax = numpy.amax(points, axis=0)
    xyrng = xymax - xymin
    xymin = xymin - xymrg*xymrg
    xymax = xymax + xymrg*xymrg
    return xymin, xymax
####################################################################
def is_inside_polygon(point, verts, edges=None, BIG=10):
    xymin, xymax = get_bounding_box(numpy.vstack([point, verts]))
    M = 2*max(numpy.hypot(xymin[0], xymin[1]), numpy.hypot(xymax[0], xymax[1]))

    if edges is None:
        edges = [[i, (i+1)%len(verts)] for i in range(len(verts))]
    
    a = 0.5*numpy.pi*numpy.random.rand()
    c = numpy.cos(a)
    s = numpy.sin(a)
    uvb = point + M*numpy.array([c, s])

    inside = False
    for e in edges:
        v1 = verts[e[0]]
        v2 = verts[e[1]]
        if (((v1[0] - point[0])*s > (v1[1] - point[1])*c) != ((v2[0] - point[0])*s > (v2[1] - point[1])*c)) and ((point[0] - v1[0])*(v1[1] - v2[1]) > (point[1] - v1[1])*(v1[0] - v2[0])) != ((uvb[0] - v1[0])*(v1[1] - v2[1]) > (uvb[1] - v1[1])*(v1[0] - v2[0])):
            inside = not inside
    return inside
####################################################################
class NestedPolygon:
    def __init__(self, ascendants=[], parent=[], children=[]):
        self.ascendants = ascendants
        self.parent = parent
        self.children = children
        return

    def level(self):
        return len(self.ascendants)
####################################################################
def make_NestingPolygonTree(paths):
    npaths = len(paths)
    BIG = 10
    NestingTree = []
    for path in paths:
        NestingTree.append(NestedPolygon(ascendants=[], parent=[], children=[]))
        for p in path:
            BIG = max(BIG, numpy.hypot(p[0], p[1]))
    BIG *= 2
    
    for i in range(npaths-1):
        for j in range(i+1,npaths):
            if is_inside_polygon(paths[j][0], paths[i],
                                 edges=None, BIG=BIG):
                NestingTree[j].ascendants.append(i)
            elif is_inside_polygon(paths[i][0], paths[j],
                                   edges=None, BIG=BIG):
                NestingTree[i].ascendants.append(j)

    maxlevel = 0
    for i in range(npaths):
        maxlevel = max(maxlevel, NestingTree[i].level())

    for level in range(maxlevel+1):
        for i in range(npaths):
            if NestingTree[i].level() == level:
                for j in NestingTree[i].ascendants:
                    if NestingTree[j].level() == level-1:
                        NestingTree[j].children.append(i)
                        NestingTree[i].parent = j
                        break
                    
    return NestingTree

