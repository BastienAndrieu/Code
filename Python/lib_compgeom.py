import numpy
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
from lib_linalg import matmul, matvecprod

#############################################################
def diff_angle(a1, a2):
    c1 = cos(a1)
    s1 = sin(a1)
    c2 = cos(a2)
    s2 = sin(a2)
    return atan2(s1*c2 - c1*s2, c1*c2 + s1*s2)
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

####################################################################
def rotation_matrix2d(angle):
    c = numpy.cos(angle)
    s = numpy.sin(angle)
    return numpy.array([[c, s], [-s, c]])

####################################################################
def minimum_area_OBB(xy):
    """
    returns (center, ranges, axes)
    """
    # get convex hull
    hull = quickhull2d(xy)

    # handle special cases
    if len(hull) < 1:
        return (numpy.zeros(2), numpy.zeros(2), numpy.eye(2))
    elif len(hull) == 1:
        return (xy[hull[0]], numpy.zeros(2), numpy.eye(2))
    elif len(hull) == 2:
        center = 0.5*numpy.sum(xy[hull], axis=0)
        vec = xy[hull[1]] - xy[hull[0]]
        ranges = numpy.array([
            0.5*numpy.hypot(vec[0], vec[1]),
            0
            ])
        axes = rotation_matrix2d(-numpy.arctan2(vec[1], vec[0]))
        return (center, ranges, axes)

    nh = len(hull)
    xyh = xy[hull]
    area = 1e20
    for i in range(nh):
        # i-th edge of the convex hull
        vec = xyh[(i+1)%nh] - xyh[i]

        # apply rotation that makes that edge parallel to the x-axis
        rot = rotation_matrix2d(numpy.arctan2(vec[1], vec[0]))
        xyrot = matmul(rot, xyh.T).T

        # xy ranges of the rotated convex hull
        mn = numpy.amin(xyrot, axis=0)
        mx = numpy.amax(xyrot, axis=0)
        ranges_tmp = mx - mn
        area_tmp = ranges_tmp[0]*ranges_tmp[1]
        
        if area_tmp < area:
            area = area_tmp
            # inverse rotation
            rot = rot.T
            center = matvecprod(rot, 0.5*(mn + mx))
            if ranges_tmp[1] > ranges_tmp[0]:
                ranges = 0.5*ranges_tmp[[1,0]]
                axes = numpy.zeros((2,2))
                axes[:,0] = rot[:,1]
                axes[:,1] = -rot[:,0]
            else:
                ranges = 0.5*ranges_tmp
                axes = rot
    return (center, ranges, axes)
####################################################################
def quickhull2d(xy):
    EPS = sys.float_info.epsilon
    nxy = len(xy)

    # handle special cases nxy = 0 and nxy = 1
    if nxy == 0:
        return []
    elif nxy == 1:
        return [0]

    # find leftmost (L) and rightmost (R) points
    L = 0
    R = 0
    for i in range(nxy):
        if xy[i][0] < xy[L][0] or (xy[i][0] < xy[L][0] + EPS and xy[i][1] < xy[L][1]):
            L = i
        elif xy[i][0] > xy[R][0] or (xy[i][0] > xy[R][0] - EPS and xy[i][1] > xy[R][1]):
            R = i

    # partition point cloud in two
    part = [i for i in range(nxy)]
    hull = [L]
    partLR = points_right_to_line(xy, part, xy[L], xy[R])
    hull = quickhull2d_partition(xy, partLR, L, R, hull)
    hull.append(R)

    partRL = points_right_to_line(xy, part, xy[R], xy[L])
    hull = quickhull2d_partition(xy, partRL, R, L, hull)
    
    return hull

##################

def quickhull2d_partition(xy, part, A, B, hull):
    if len(part) < 1:
        return hull

    # find point C at maximal distance to the right of line [AB]
    # compute components of a normal to and pointing to the right of line [AB]
    xn = xy[B][1] - xy[A][1]
    yn = xy[A][0] - xy[B][0]
    distC = -1.0
    for P in part:
        if P == A or P == B:
            continue
        distP = (xy[P][0] - xy[A][0])*xn + (xy[P][1] - xy[A][1])*yn
        if distP > distC:
            C = P
            distC = distP
            
    # recursion on new partitions
    # 1) points right to line [AC]
    partAC = points_right_to_line(xy, part, xy[A], xy[C])
    hull = quickhull2d_partition(xy, partAC, A, C, hull)
    hull.append(C)
    
    # 2) points right to line [CB]
    partCB = points_right_to_line(xy, part, xy[C], xy[B])
    hull = quickhull2d_partition(xy, partCB, C, B, hull)
    
    return hull

##################

def points_right_to_line(xy, part, p1, p2):
    points = []
    xn = p2[1] - p1[1]
    yn = p1[0] - p2[0]
    for P in part:
        dist = (xy[P][0] - p1[0])*xn + (xy[P][1] - p1[1])*yn
        if dist > 0:
            points.append(P)
    return points
