import numpy
import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_halfedge as lhe


def distance_from_segment(segment, xy):
    # segment [[xa, ya], [xb, yb]]
    t = [segment[1][i] - segment[0][i] for i in range(2)]
    tsqr = t[0]*t[0] + t[1]*t[1]
    v = [xy[i] - segment[0][i] for i in range(2)]
    v_dot_t = v[0]*t[0] + v[1]*t[1]
    #proj = []
    #print 'v_dot_t = ', v_dot_t, ', t_dot_t =', tsqr
    if v_dot_t < 0:
        #print 'v_dot_t < 0'
        proj = segment[0]
    elif v_dot_t > tsqr:
        #print 'v_dot_t > t_dot_t'
        proj = segment[1]
    else:
        #print '0 < v_dot_t < t_dot_t'
        v_dot_t = v_dot_t/tsqr # 0 < * < 1
        proj = [segment[0][i] + v_dot_t*t[i] for i in range(2)]
        #dist = v[0]*v[0] + v[1]*v[1] - v_dot_t*v_dot_t
    #else:
    dist = 0
    for i in range(2):
        dist += (xy[i] - proj[i])*(xy[i] - proj[i])
    #print 'xy = ', xy, ', segment = ', [[x for x in p] for p in segment], ', proj =', proj, ', dist = ', numpy.sqrt(dist)
    return dist, proj
    











# READ INPUT DATA
print 'read data...'
pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
iface = 2
strf = format(iface,'03')
tri = numpy.loadtxt(pth + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pth + 'brepmesh/xyz_' + strf + '.dat', dtype=float)
xyz = xyz[:,[1,2,0]]
print '   ok.'



# MAKE HALFEDGE DS
print 'make halfedge DS...'
mesh = lhe.pydata_to_SurfaceMesh(xyz[:,0:2], tri)
print '   ok.'



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
                








# VISU
print 'plot...'
fig, ax = plt.subplots()
mesh.plot_as_triangulation(ax, color='k')
#
if False:
    #ax.plot(cell_centers[bound_cells,0], cell_centers[bound_cells,1], 'r.')
    for c in range(n_cells):#bound_cells:
        if distance[c] > 100: continue
        ax.plot(
            [cell_centers[c][0], closest_point[c][0]],
            [cell_centers[c][1], closest_point[c][1]],
            'r-'
        )
else:
    x = [v.co[0] for v in mesh.verts]
    y = [v.co[1] for v in mesh.verts]
    tri = [f[0:3] for f in mesh.f2v]
    ax.tripcolor(x, y, tri, numpy.sqrt(distance))
#
print '   ok.'
ax.set_aspect('equal')
plt.show()
