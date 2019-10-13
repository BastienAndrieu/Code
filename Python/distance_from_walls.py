# -*-coding:Latin-1 -*
ROOT = '/home/bastien/'#'/d/bandrieu'

import numpy
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import meshio
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.append(ROOT+'GitHub/Code/Python')
import lib_halfedge as lhe


def distance_from_segment(segment, xy):
    # segment [[xa, ya], [xb, yb]]
    t = [segment[1][i] - segment[0][i] for i in range(2)]
    tsqr = t[0]*t[0] + t[1]*t[1]
    v = [xy[i] - segment[0][i] for i in range(2)]
    v_dot_t = v[0]*t[0] + v[1]*t[1]
    if v_dot_t < 0:
        proj = segment[0]
    elif v_dot_t > tsqr:
        proj = segment[1]
    else:
        v_dot_t = v_dot_t/tsqr # 0 < * < 1
        proj = [segment[0][i] + v_dot_t*t[i] for i in range(2)]
    #
    dist = 0
    for i in range(2):
        dist += (xy[i] - proj[i])*(xy[i] - proj[i])
    return dist, proj
    











# READ INPUT DATA
print 'read data...'
if True:
    """
    pth = ROOT+'GitHub/FFTsurf/test/demo_EoS_brep/'
    ifaces = [2]#,7]
    xy = numpy.empty((0,2))
    f2v = []
    for iface in ifaces:
        strf = format(iface,'03')
        tri = numpy.loadtxt(pth + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1 + len(xy)
        xyz = numpy.loadtxt(pth + 'brepmesh/xyz_' + strf + '.dat', dtype=float)
        for t in tri:
            f2v.append([int(v) for v in t])
        xy = numpy.vstack([xy, xyz[:,[1,2]]])
    """
    pth = ROOT+'Téléchargements/ne_50m_admin/'
    land = 'bolivia_mali_iceland'
    xy = numpy.loadtxt(pth+land+'_xy.dat')
    f2v = numpy.loadtxt(pth+land+'_tri.dat', dtype=int)
    
else:
    pth = ROOT+'Téléchargements/mesquite-2.3.0/meshFiles/2D/vtk/'
    filename = pth+'tris/untangled/tri_20258.vtk'#mixed/untangled/overt_hyb_2.vtk'#N-Polygonal/poly3.vtk'#tris/untangled/bad_circle_tri.vtk'#
    if True:
        reader = vtk.vtkUnstructuredGridReader()#vtkPolyDataReader()#
        reader.SetFileName(filename)
        reader.Update()
        data = reader.GetOutput()
        xyz = vtk_to_numpy(data.GetPoints().GetData())
        xy = xyz[:,0:2]
        f2v = []
        used = numpy.zeros(len(xy), dtype=bool)
        for i in range(data.GetNumberOfCells()):
            cell = data.GetCell(i)
            f = []
            for j in range(cell.GetNumberOfPoints()):
                v = int(cell.GetPointId(j))
                used[v] = True
                f.append(v)
            f2v.append(f)
        # remove unused vertices
        if not all(used):
            nv = 0
            v2v_renum = -numpy.ones(len(xy), dtype=int)
            for i in range(len(xy)):
                if used[i]:
                    v2v_renum[i] = nv
                    nv += 1
            for i in range(len(f2v)):
                for j in range(len(f2v[i])):
                    f2v[i][j] = int(v2v_renum[f2v[i][j]])
            xy = [xy[i] for i in range(len(xy)) if used[i]]
    else:
        m = meshio.read(filename)
        xy = numpy.array([p[0:2] for p in m.points])
        f2v = []
        for key, values in m.cells.iteritems():
            f2v.extend(values)
print '   ok.'



if False:
    fig, ax = plt.subplots()
    for f in f2v:
        v = f[:] + [f[0]]
        x = [xy[i][0] for i in v]
        y = [xy[i][1] for i in v]
        ax.plot(x, y, 'k-')
    ax.set_aspect('equal')
    plt.show()



# MAKE HALFEDGE DS
print 'make halfedge DS...'
mesh = lhe.pydata_to_SurfaceMesh(xy, f2v)
print '   ok.'

if False:
    print [v.edge for v in mesh.verts]
    fig, ax = plt.subplots()
    lhe.plot_mesh(mesh,
                  ax,
                  faces=False,
                  edges=True,
                  halfedges=False,
                  vertices=False,
                  boundaries=False,
                  v2h=True,
                  v2f=False,
                  count_from_1=False)
    ax.set_aspect('equal')
    plt.show()
    


# GET CELL CENTERS
print 'compute cell centers...'
cell_centers = numpy.zeros((len(mesh.f2v),2))
for i in range(len(mesh.f2v)):
    invn = 1./float(len(mesh.f2v[i]))
    for v in mesh.f2v[i]:
        for k in range(2):
            cell_centers[i][k] = cell_centers[i][k] + mesh.verts[v].co[k]*invn
print '   ok.'



# GET BOUNDARY CELLS
print 'get boundary cells...'
boundaries = mesh.get_boundaries()
bound_cells = []
for b in boundaries:
    cells = [mesh.get_face(e) for e in b]
    for c in cells:
        if c not in bound_cells: bound_cells.append(c)
print '   ok.'



# GET "SURFACE" MESH
print 'get surface mesh...'
SM_verts_xy = []
SM_e2v = []
v2v_renum = [-1 for v in mesh.verts]
for b in boundaries:
    for e in b:
        verts = [mesh.get_orig(e), mesh.get_dest(e)]
        for v in verts:
            if v2v_renum[v] < 0:
                v2v_renum[v] = len(SM_verts_xy)
                SM_verts_xy.append(mesh.verts[v].co)
        SM_e2v.append([v2v_renum[v] for v in verts])
print '   ok.'



# COMPUTE DISTANCES AND PROJECTIONS (BRUTE FORCE)
print 'compute boundary cell centers -> surface distances...'
n_cells = len(mesh.f2v)
closest_point = numpy.zeros((n_cells,2))
distance = 1e20*numpy.ones(n_cells)
closest_edge = -1*numpy.ones(n_cells, dtype=int)
for c in bound_cells:
    for i, e in enumerate(SM_e2v):
        dist, proj = distance_from_segment(
            segment=[SM_verts_xy[v] for v in e],
            xy=cell_centers[c]
        )
        if dist < distance[c]:
            distance[c] = dist
            closest_point[c] = proj[:]
            closest_edge[c] = i

print '   ok.'



# INITIALIZE QUEUE
print 'initialize queue...'
from collections import deque
#in_queue = [0 for c in mesh.f2v] # -1: boundary cell, 0: outside the queue, 1: inside the queue
in_queue = numpy.zeros(n_cells, dtype=int)
queue = deque([])
for c in bound_cells:
    in_queue[c] = -1
    n = len(mesh.f2v[c])
    # push neighbor cells in queue
    for i in range(n):
        e = mesh.get_twin([c,i])
        if e is None: continue
        j = mesh.get_face(e)
        if j > 0:
            if in_queue[j] == 0:
                in_queue[j] = 1
                queue.append(j)
print '   ok.'

print 'compute interior distances...'
# PROCESS QUEUE
while len(queue) > 0:
    cur_cell = queue.popleft()
    if in_queue[cur_cell] == -1: continue # should not happen
    in_queue[cur_cell] = 0
    n = len(mesh.f2v[cur_cell])
    # loop over cells adjacent to current cell (aka neighbors)
    icell_src = None
    for k in range(n):
        icell1 = mesh.get_face(mesh.get_twin([cur_cell,k])) # get k-th neighbor cell
        if closest_edge[cur_cell] != closest_edge[icell1]:
            dist, proj = distance_from_segment(
                segment=[SM_verts_xy[v] for v in SM_e2v[closest_edge[icell1]]],
                xy=cell_centers[cur_cell]
            )
            if dist < distance[cur_cell]:
                icell_src = icell1
                distance[cur_cell] = dist
                closest_point[cur_cell] = proj[:]
                closest_edge[cur_cell] = closest_edge[icell1]
    #
    # update distance and add neighbors cell into the queue
    if icell_src is not None:
        for k in range(n):
            icell1 = mesh.get_face(mesh.get_twin([cur_cell,k])) # get k-th neighbor cell
            if icell1 == icell_src: continue
            if in_queue[icell1] == 0:
                in_queue[icell1] = 1
                queue.append(icell1)
print '   ok.'
    








# VISU
print 'plot...'

all_tri = all([len(f) == 3 for f in mesh.f2v])

fig, ax = plt.subplots()
"""
if all_tri:
    mesh.plot_as_triangulation(ax, color='k')
else:
    lhe.plot_mesh(mesh,
                  ax,
                  faces=False,
                  edges=True,
                  halfedges=False,
                  vertices=False,
                  boundaries=False,
                  v2h=False,
                  v2f=False,
                  count_from_1=False)
"""
#
x = [v.co[0] for v in mesh.verts]
y = [v.co[1] for v in mesh.verts]
if not all_tri:
    #ax.plot(cell_centers[bound_cells,0], cell_centers[bound_cells,1], 'r.')
    """
    for c in bound_cells:#range(n_cells):#
        if distance[c] > 100: continue
        ax.plot(
            [cell_centers[c][0], closest_point[c][0]],
            [cell_centers[c][1], closest_point[c][1]],
            'r-'
        )
        ax.add_artist(
            plt.Circle(cell_centers[c], numpy.sqrt(distance[c]), color='r')
        )
    """
    tri = []
    d = []
    for i, f in enumerate(mesh.f2v):
        n = len(f)
        if n == 3:
            tri.append(f)
            d.append(distance[i])
        else:
            for j in range(1,n-1):
                tri.append([f[0], f[j], f[j+1]])
                d.append(distance[i])
else:
    tri = [f[0:3] for f in mesh.f2v]
    d = distance
ax.tripcolor(x, y, tri, numpy.sqrt(d), cmap=cm.YlOrRd)
#
if all_tri:
    mesh.plot_as_triangulation(ax, color='k')
else:
    lhe.plot_mesh(mesh,
                  ax,
                  faces=False,
                  edges=True,
                  halfedges=False,
                  vertices=False,
                  boundaries=False,
                  v2h=False,
                  v2f=False,
                  count_from_1=False)
#
print '   ok.'
ax.set_aspect('equal')
plt.show()
