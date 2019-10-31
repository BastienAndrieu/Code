import numpy
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
    #
    def get_prev(self, ihedg):
        n = len(self.f2v[ihedg[0]])
        return [ihedg[0], (ihedg[1]+n-1)%n]
    #
    def get_next(self, ihedg):
        n = len(self.f2v[ihedg[0]])
        return [ihedg[0], (ihedg[1]+1)%n]
    #
    def get_face(self, ihedg):
        return ihedg[0]
    #
    def get_twin(self, ihedg):
        return self.twin[ihedg[0]][ihedg[1]]
    #
    def get_orig(self, ihedg):
        return self.f2v[ihedg[0]][ihedg[1]]
    #
    def get_dest(self, ihedg):
        return self.get_orig(self.get_next(ihedg))
    #
    def is_boundary_edge(self, ihedg):
        return self.get_twin(ihedg) == None
    #
    def get_boundaries(self):
        boundaries = []
        visited = [False for startv in self.verts]
        for startv in self.verts:
            if visited[startv.index]: continue
            visited[startv.index] = True
            if not self.is_boundary_edge(startv.edge): continue
            edges = [startv.edge]
            while True:
                v = self.get_dest(edges[-1])
                visited[v] = True
                e = self.verts[v].edge
                if e == edges[0]: break
                edges.append(e)
            boundaries.append(edges)
        return boundaries
    #
    def plot_as_triangulation(self, ax, linewidth=1, color='b'):
        import matplotlib.pyplot as plt
        x = [v.co[0] for v in self.verts]
        y = [v.co[1] for v in self.verts]
        tri = [f[0:3] for f in self.f2v]
        return plt.triplot(x, y, tri, lw=linewidth, color=color)
############################################################

def hash_integer_pair(p, q):
    r = min(p, q)
    if p == q or r < 0:
        return -1
    else:
        s = max(p, q)
        return int(r + s*(s-1)/2)


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

def plot_mesh(mesh,
              ax,
              faces=True,
              edges=True,
              halfedges=True,
              vertices=True,
              boundaries=True,
              v2h=True,
              v2f=False,
              count_from_1=True):
    import numpy
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection
    if count_from_1:
        offset = 1
    else:
        offset = 0
        
    # vertex labels
    if vertices:
        for v in mesh.verts:
            txt = ax.text(v.co[0], v.co[1], str(v.index+offset),
                          zorder=10)
            txt.set_bbox(dict(facecolor='y', alpha=0.5, edgecolor='y'))
    
    # face centroids
    fctr = numpy.zeros((len(mesh.f2v), 2))
    for i in range(len(mesh.f2v)):
        f = mesh.f2v[i]
        for v in f:
            fctr[i] = fctr[i] + mesh.verts[v].co[0:2]
        fctr[i] = fctr[i]/float(len(f))

    # plot faces
    if faces:
        patches = []
        for i, f in enumerate(mesh.f2v):
            ax.text(fctr[i,0], fctr[i,1], str(i+offset))
            polygon = numpy.array([mesh.verts[v].co[0:2] for v in f])
            patches.append(Polygon(polygon, True))
        colors = 100*numpy.arange(len(mesh.f2v))
        p = PatchCollection(patches, alpha=0.2, edgecolors=None)
        p.set_array(numpy.array(colors))
        ax.add_collection(p)
    
    # plot edges
    for i in range(len(mesh.f2v)):
        fi = mesh.f2v[i]
        for j in range(len(fi)):
            v1 = mesh.verts[get_orig([i,j], mesh)]
            v2 = mesh.verts[get_dest([i,j], mesh)]
            if is_boundary_edge([i,j], mesh):
                if halfedges:
                    xym = 0.5*(v1.co + v2.co)
                    x = [fctr[i,0], xym[0]]
                    y = [fctr[i,1], xym[1]]
                    ax.plot(x, y, 'b-', lw=0.5)
                if edges: ax.plot([v1.co[0], v2.co[0]], [v1.co[1], v2.co[1]], 'k', lw=0.8)
            else:
                ej = get_twin([i,j], mesh)
                if edges:
                    if v1.index < v2.index:
                        ax.plot([v1.co[0], v2.co[0]], [v1.co[1], v2.co[1]], 'k', lw=0.8)
                if halfedges:
                    ax.plot([fctr[i,0], fctr[ej[0],0]], [fctr[i,1], fctr[ej[0],1]],
                            'g-', lw=0.5)
    # boundaries
    if boundaries:
        for loop in get_boundary_loops(mesh):
            lenloop = len(loop)
            for i in range(lenloop):
                j = (i+1)%lenloop
                x = [mesh.verts[k].co[0] for k in [i,j]]
                y = [mesh.verts[k].co[1] for k in [i,j]]
                ax.plot(x, y, 'b-', lw=1.0)

    # vertex to incident halfedge
    if v2h:
        for v in mesh.verts:
            if not is_boundary_edge(v.edge, mesh): continue
            w = mesh.verts[get_dest(v.edge, mesh)]
            vec = 0.5*(w.co - v.co)
            ax.quiver(v.co[0], v.co[1], vec[0], vec[1],
                      color='r', lw=1.5, edgecolor='r',
                      scale_units='xy', scale=1, zorder=2)
    if v2f:
        for v in mesh.verts:
            xy = []
            e = v.edge
            n = 0
            if is_boundary_edge(e, mesh):
                w = mesh.verts[get_dest(e, mesh)]
                xy.append(0.5*(w.co[0:2] + v.co[0:2]))
                n += 1
            while True:
                xy.append(fctr[e[0]])
                n += 1
                e = get_prev(e, mesh)
                if is_boundary_edge(e, mesh):
                    w = mesh.verts[get_orig(e, mesh)]
                    xy.append(0.5*(w.co[0:2] + v.co[0:2]))
                    break
                e = get_twin(e, mesh)
                if e[0] == v.edge[0]:
                    break
            #
            for i in range(n):
                xyi = 0.5*(xy[i] + v.co[0:2])
                xyj = 0.5*(xy[(i+1)%len(xy)] + v.co[0:2])
                ax.quiver(xyi[0], xyi[1], xyj[0]-xyi[0], xyj[1]-xyi[1],
                          color='c', lw=0.1, edgecolor='c',
                          scale_units='xy', scale=1, zorder=3)
    return


#######################################################
def barycentric_coords(poly, point):
    return None

#######################################################
def locate_point_2d(mesh, point, iface=None):
    if iface is None:
        iface = numpy.random.randint(len(mesh.f2v))
    #
    jface = iface
    visited_faces = []
    while True:
        if jface in visited_faces:
            break
        visited_faces.append(jface)
        #
        #
        verts = mesh.f2v[jface]
        m = len(verts)
        inside = True
        for i in range(m):
            u = mesh.verts[verts[(i+1)%m]].co - mesh.verts[verts[i]].co
            v = point - mesh.verts[verts[i]].co
            if u[0]*v[1] < u[1]*v[0]:
                inside = False
                twin = mesh.get_twin([jface,i])
                if mesh.is_boundary_edge(twin):
                    return False, jface, None
                jface = mesh.get_face(twin)
                break
        #
        if inside:
            poly = [mesh.verts[v].co for v in verts]
            return jface, barycentric_coords(poly, point)
    
