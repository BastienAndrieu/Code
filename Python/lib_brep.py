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
def make_wire(head, halfedges, curves, vexclude=[]):
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
