############################################################
class Vertex:
    def __init__(self, co, edge, index):
        self.co = co
        self.edge = edge
        self.index = index
###########################
class Face:
    def __init__(self, verts, edges, index):
        self.verts = verts
        self.edges = edges
        self.index = index
###########################
class Halfedge:
    def __init__(self, face, twin, prev, next, orig):
        self.face = face
        self.twin = twin
        self.prev = prev
        self.next = next
        self.orig = orig
###########################
class HEmesh:
    def __init__(self, verts, edges, faces):
        self.verts = verts
        self.edges = edges
        self.faces = faces
###########################
class SurfaceMesh:
    def __init__(self, verts, f2v, twin):
        self.verts = verts
        self.f2v = f2v
        self.twin = twin
############################################################

def hash_integer_pair(p, q):
    r = min(p, q)
    if p == q or r < 0:
        return -1
    else:
        s = max(p, q)
        return r + s*(s-1)/2


"""
def pydata_to_HEmesh(vco, f2v):
    mesh = HEmesh(verts=[], edges=[], faces=[])

    halfedges = {}

    for x in vco:
        mesh.verts.append(Vertex(co=x, edge=None]))
        
    for face in f2v:
        mesh.faces.append(Face(verts=[mesh.verts[ivert] for ivert in face], edge=None))
        
    for iface, face in enumerate(f2v):
        nv = len(face)
        for iedge in range(nv):
            vpair = [face[iedge], face[(iedge+1)%nv]]
            ih = hash_integer_pair(vpair[0], vpair[1])
            if ih not in halfedges:
                halfedges[ih] = [iface, iedge]
            else:
                mesh.verts.append(Halfedge(face, twin, prev, next, orig)
                #    mesh.faces.append(Face(verts=[mesh.verts[i] for i in f], edge=None))
"""

def pydata_to_SurfaceMesh(vco, f2v):
    mesh = SurfaceMesh(verts=[], f2v=f2v, twin=[])

    for i, x in enumerate(vco):
        mesh.verts.append(Vertex(co=x, edge=None, index=i))

    for face in f2v:
        ned = len(face)
        mesh.twin.append([None]*ned)

    halfedges = {}
    for ifa, face in enumerate(f2v):
        ned = len(face)
        for ied in range(ned):
            pairv = [face[ied], face[(ied+1)%ned]]
            ih = hash_integer_pair(pairv[0], pairv[1])
            if ih in halfedges:
                # collision
                mesh.twin[ifa][ied] = halfedges[ih]
                mesh.twin[halfedges[ih][0]][halfedges[ih][1]] = [ifa, ied]
            else:
                halfedges[ih] = [ifa, ied]

    # vertex -> outgoing half-edge
    for ifa, face in enumerate(f2v):
        ned = len(face)
        for ied in range(ned):
            iv = face[ied]
            if mesh.twin[ifa][ied] != None:
                if mesh.verts[iv].edge != None: continue
            
            mesh.verts[iv].edge = [ifa,ied]
    return mesh


"""
def pydata_to_HEmesh(vco, f2v):
    mesh = HEmesh(verts=[], faces=[])

    for x in vco:
        mesh.verts.append(Vertex(co=x, edge=None]))

    for face in f2v:
        ned = len(face)
        mesh.faces.append(Face(verts=[mesh.verts[iv] for iv in face], edges=[None]*ned))

    halfedges = {}
    for ifa, face in enumerate(f2v):
        ned = len(face)
        for ied in range(ned):
            pairv = [face[ied], face[(ied+1)%ned]]
            ih = hash_integer_pair(pairv[0], pairv[1])
            if ih in halfedges:
                # collision
                
                #mesh.twin.[ifa][ied] = halfedges[ih]
                #mesh.twin.[halfedges[ih][0]][halfedges[ih][1]] = [ifa, ied]
            else:
                halfedges[ih] = [ifa, ied]

    # vertex -> outgoing half-edge
    for ifa, face in enumerate(f2v):
        ned = len(face)
        for ied in range(ned):
            iv = face[ied]
            if mesh.twin[ifa][ied] != None:
                if mesh.verts[iv].edge != None: continue
            
            mesh.verts[iv].edge = [ifa,ied]
    return mesh
"""

##############################################################################
def get_prev(ihedg, mesh):
    n = len(mesh.f2v[ihedg[0]])
    return [ihedg[0], (ihedg[1]+n-1)%n]
#######################################
def get_next(ihedg, mesh):
    n = len(mesh.f2v[ihedg[0]])
    return [ihedg[0], (ihedg[1]+1)%n]
#######################################
def get_face(ihedg):
    return ihedg[0]
#######################################
def get_twin(ihedg, mesh):
    return mesh.twin[ihedg[0]][ihedg[1]]
#######################################
def get_orig(ihedg, mesh):
    return mesh.f2v[ihedg[0]][ihedg[1]]
#######################################
def get_dest(ihedg, mesh):
    return get_orig(get_next(ihedg, mesh), mesh)
#######################################
def is_boundary_edge(ihedg, mesh):
    return mesh.twin[ihedg[0]][ihedg[1]] == None
#######################################
def get_boundary_loops(mesh):
    # extract all boundary loops
    boundary_loops = []
    visited = [False for startv in mesh.verts]
    for startv in mesh.verts:
        if visited[startv.index]: continue
        visited[startv.index] = True
        if not is_boundary_edge(startv.edge, mesh): continue
        loop = []
        loop.append(startv.index)
        while True:
            v = mesh.verts[loop[-1]]
            e = v.edge
            nextv = get_dest(e, mesh)
            if nextv == loop[0]: break
            visited[nextv] = True
            loop.append(nextv)
        boundary_loops.append(loop)
    return boundary_loops
#######################################
def get_v2f(mesh, v):
    e = v.edge
    faces = []
    while True:
        f = e[0]
        faces.append(f)
        e = get_prev(e, mesh)
        if is_boundary_edge(e, mesh): return faces
        e = get_twin(e, mesh)
        if e[0] == v.edge[0]: return faces
#######################################
def get_v2v(mesh, v):
    e = v.edge
    verts = []
    while True:
        w = mesh.verts[get_dest(e, mesh)]
        verts.append(w)
        e = get_prev(e, mesh)
        if is_boundary_edge(e, mesh):
            w = mesh.verts[get_orig(e, mesh)]
            verts.append(w)
            return verts
        e = get_twin(e, mesh)
        if e[0] == v.edge[0]: return verts
#######################################
