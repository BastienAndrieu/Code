import numpy

############################################
class BRep:
    def __init__(self, verts, edges, curves, faces):
        self.verts = verts
        self.edges = edges
        self.faces = faces
        return
############################################
class Face:
    def __init__(self, outer, inner, index):
        self.outer = outer
        self.inner = inner
        self.index = index
        return
    ###################
    def get_polygon(self):
        polys = []
        polys.append(self.outer.get_polygon())
        for wire in self.inner:
            polys.append(wire.get_polygon())
        return polys
############################################
class Wire:
    def __init__(self, edges):
        self.edges = edges
        return
    ###################
    def reverse(self):
        for edge in self.edges:
            edge.reverse()
        self.edges = self.edges[::-1]
        return
    ###################
    def copy(self):
        return Wire(edges=[e.copy() for e in self.edges])
    ###################
    def reverse_copy(self):
        w = self.copy()
        w.reverse()
        return w
    ###################
    def get_polygon(self):
        poly = []
        for edge in self.edges:
            for point in edge.uv[:-1]:
                poly.append(point)
        return poly
############################################
class Halfedge:
    def __init__(self, face, orig, twin, prev, next, ihyp):
        self.face = int(face)
        self.orig = int(orig)
        self.twin = int(twin)
        self.prev = int(prev)
        self.next = int(next)
        self.ihyp = int(ihyp)
        return
############################################
class Vertex:
    def __init__(self, xyz, edge, index):
        self.xyz = xyz
        self.edge = edge
        self.index = index
        return
############################################
class Curve:
    def __init__(self, xyz, uv):
        self.xyz = xyz
        self.uv = uv
        return
    ###################
    def reverse(self):
        self.xyz = self.xyz[::-1]
        self.uv = self.uv[::-1]
        return
    ###################
    def copy(self):
        return Curve(xyz=self.xyz[:], uv=self.uv[:])
############################################


############################################
def make_wire(head, halfedges, curves, vexclude=None):
    if vexclude is None:
        vexclude = []
    wedges = []
    ih = head
    while True:
        if halfedges[ih].orig not in vexclude:
            start = ih
            break
        ih = halfedges[ih].next
        if ih == head: break
    while True:
        ic = int(ih/2)
        if ih%2 == 0:
            wedges.append(Curve(xyz=curves[ic].xyz[:],
                                uv=curves[ic].uv[:,2:4]))
        else:
            wedges.append(Curve(xyz=curves[ic].xyz[::-1],
                                uv=curves[ic].uv[::-1,0:2]))
        ih = halfedges[ih].next
        if ih == start:
            return Wire(edges=wedges)
############################################

############################################
def read_halfedges(filename):
    h = numpy.loadtxt(filename, dtype=int) - 1
    halfedges = []
    for i, e in enumerate(h):
        if i%2 == 0:
            twinh = i + 1
        else:
            twinh = i - 1
        prevh = 2*e[2] + e[3]
        nexth = 2*e[4] + e[5]
        halfedges.append(
            Halfedge(
                face=e[0],
                orig=e[1],
                twin=twinh,
                prev=prevh,
                next=nexth,
                ihyp=-1
                )
        )
    return halfedges
############################################

############################################
def read_curves(fileuv, filexyz):
    fu = open(fileuv, 'r')
    fx = open(filexyz, 'r')
    ncurves_u = int(fu.readline())
    ncurves_x = int(fx.readline())
    ncurves = min(ncurves_u, ncurves_x)
    curves = []
    for icurv in range(ncurves):
        npoints = int(fx.readline())
        fu.readline()
        xyz = numpy.zeros((npoints,3))
        uv = numpy.zeros((npoints,4))
        for i in range(npoints):
            xyz[i] = [float(a) for a in fx.readline().split()]
            uv[i]  = [float(a) for a in fu.readline().split()]
        curves.append(
            Curve(
                xyz=xyz,
                uv=uv
            )
        )
    fu.close()
    fx.close()
    return curves
############################################



############################################
def read_faces(filename, halfedges, curves):
    f = open(filename, 'r')
    faces = []

    nf = 0
    for h in halfedges:
        nf = max(nf, h.face)
    nf += 1

    jf = 0
    while jf < nf:
        he = [int(a)-1 for a in f.readline().split()]
        ih = 2*he[0] + he[1]
        wout = make_wire(ih, halfedges, curves)
        winn = []
        ninner = int(f.readline())
        for i in range(ninner):
            he = [int(a)-1 for a in f.readline().split()]
            ih = 2*he[0] + he[1]
            winn.append(make_wire(ih, halfedges, curves))
        faces.append(
            Face(
                outer=wout,
                inner=winn,
                index=jf
            )
        )
        jf += 1
    f.close()
    return faces
############################################
