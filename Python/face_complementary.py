import math
import numpy
import os
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
from lib_brep import *
########################################################

########################################################
def sample(array, nsample=3):
    n = len(array)
    if nsample > n:
        if n == 2:
            points = [array[0]]
            for i in range(1,nsample-1):
                t1i = i/float(nsample-1)
                t0i = 1.0 - t1i
                pi = [t0i*array[0][0] + t1i*array[1][0], t0i*array[0][1] + t1i*array[1][1]]
                points.append(pi)
            points.append(array[1])
            return points
        else:
            nsample = min(n, nsample)
            return [array[int(i*(n-1)/float(nsample-1))] for i in range(nsample)]
    else:
        return [array[int(i*(n-1)/float(nsample-1))] for i in range(nsample)]
########################################################
def adaptive_polyline(endpoints, resolution=0.025):
    dist = math.sqrt((endpoints[0][0] - endpoints[1][0])**2 + (endpoints[0][1] - endpoints[1][1])**2)
    n = 1 + int(math.ceil(dist/resolution))
    return sample(endpoints, nsample=n)
########################################################
def is_on_square_boundary(point, tol=1e-7):
    for co in point:
        if abs(abs(co) - 1.0) < tol:
            return True
    return False
########################################################
def get_border(point, tol=1e-7):
    borders = []
    if abs(point[1] + 1.0) < tol:
        borders.append(0)
    if abs(point[0] - 1.0) < tol:
        borders.append(1)
    if abs(point[1] - 1.0) < tol:
        borders.append(2)
    if abs(point[0] + 1.0) < tol:
        borders.append(3)
    return borders
########################################################
def get_corner_point(i):
    if i == 0:
        return [-1,-1]
    elif i == 1:
        return [1,-1]
    elif i == 2:
        return [1,1]
    else:
        return [-1,1]
########################################################
def get_complementary(face):
    cfaces = []
    # process inner wires
    for wire in face.inner:
        cfaces.append(Face(outer=wire.reverse_copy(), inner=[], index=-1))

    # process outer wire
    wire = face.outer.reverse_copy()
    # discard edges which lie entirely on the square
    discard = []
    for edge in wire.edges:
        isbp = [is_on_square_boundary(p) for p in sample(edge.uv, nsample=4)]
        if all(isbp):
            discard.append(edge)
    if len(discard) > 0:
        for edge in discard:
            wire.edges.remove(edge)
    else:
        wedges = []
        for iborder in range(4):
            orig = get_corner_point(iborder)
            dest = get_corner_point((iborder+1)%4)
            wedges.append(Curve(xyz=[],
                                uv=adaptive_polyline([orig, dest])))
        cfaces.append(Face(outer=Wire(edges=wedges), inner=[wire.reverse_copy()], index=-1))
        return cfaces    
    #
    edges = wire.edges
    while len(edges) > 0:
        nedges = len(edges)
        for iedge in range(nedges):
            isbp = [is_on_square_boundary(p) for p in sample(edges[iedge].uv, nsample=2)]
            if isbp[0]:
                # start new wire (face)
                discard = []
                wedges = []
                wverts = [edges[iedge].uv[0]]
                borders0 = get_border(wverts[0])
                jedge = iedge
                # concatenate edges between borders
                while True:
                    discard.append(edges[jedge])
                    wedges.append(edges[jedge].copy())
                    wverts.append(edges[jedge].uv[-1])
                    if isbp[1]:
                        # a border has been reached
                        borders1 = get_border(wverts[-1])
                        break
                    jedge += 1
                    isbp = [is_on_square_boundary(p) for p in sample(edges[jedge].uv, nsample=2)]
            #
            if len(borders0) == 1 and len(borders1) == 1:
                if borders0[0] == borders1[0]:
                    # both ends on same border 
                    wedges.append(polyline=[wverts[-1], wverts[0]])
                else:
                    # connect both ends with new edges along borders
                    if borders1[0] < borders0[0]:
                        loopborders = [i for i in range(borders1[0],borders0[0]+1)]
                    else:
                        loopborders = [i for i in range(borders1[0],4)]
                        loopborders.extend([i for i in range(borders0[0]+1)])

                    for j, i in enumerate(loopborders):
                        if j < len(loopborders)-1:
                            # add intermediary corner point
                            wverts.append(get_corner_point((i+1)%4))
                            wedges.append(Curve(xyz=[],
                                                uv=adaptive_polyline([wverts[-2], wverts[-1]])))
                        else:
                            wedges.append(Curve(xyz=[],
                                                uv=adaptive_polyline([wverts[-1], wverts[0]])))
                # add wire (face)
                cfaces.append(Face(outer=Wire(edges=wedges), inner=[], index=-1))
                for edge in discard:
                    edges.remove(edge)
            break
    return cfaces  
#################################################
def run_meshgen(filecoefs, filepoints, fileedges, fileinfo, filetri, fileuv, filexyz):
    # call system
    print 'meshgen...'
    cmd = '/stck/bandrieu/Bureau/MeshGen/./meshgen.out '
    cmd += filecoefs + ' '
    cmd += filepoints + ' '
    cmd += fileedges + ' '
    cmd += fileinfo + ' '
    cmd += filetri + ' '
    cmd += fileuv + ' '
    cmd += filexyz
    os.system(cmd)
    print 'done.'

    # get output as pydata
    tri = numpy.loadtxt(filetri, dtype=int) - 1
    uv = numpy.loadtxt(fileuv, dtype=float)
    xyz = numpy.loadtxt(filexyz, dtype=float)    
    
    return tri, uv, xyz
#################################################


pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'


args = sys.argv
if len(args) < 2:
    iface = 2
else:
    iface = int(args[1])
    
if len(args) < 3:
    EoB = False
else:
    EoB = (int(args[2]) > 0)

strf = format(iface,'03')

if EoB:
    suf = '_new'
    suf2 = '_eos'
else:
    suf = ''
    suf2 = ''

# Halfedges ###
h = numpy.loadtxt(pthin + 'debug/edges'+suf+'.dat', dtype=int) - 1
nh = len(h)
edges = []
nf = 0
for i, e in enumerate(h):
    nf = max(nf, e[0]+1)
    if i%2 == 0:
        twin = i+1
    else:
        twin = i-1
    prev = 2*e[2] + e[3]
    next = 2*e[4] + e[5]
    edges.append(Halfedge(face=e[0],
                          orig=e[1],
                          twin=twin,
                          prev=prev,
                          next=next,
                          ihyp=-1))


# Curves ###
fx = open(pthin + 'debug/edges_xyz'+suf+'.dat','r')
fu = open(pthin + 'debug/edges_uv'+suf+'.dat','r')
ne = int(fx.readline())
fu.readline()
curves = []
for ie in range(ne):
    np = int(fx.readline())
    fu.readline()
    xyz = numpy.zeros((np,3))
    uv = numpy.zeros((np,4))
    for i in range(np):
        xyz[i] = [float(a) for a in fx.readline().split()]
        uv[i]  = [float(a) for a in fu.readline().split()]
    curves.append(Curve(xyz=xyz,
                        uv=uv))
fx.close()






    
# Faces, Wires ###
f = open(pthin + 'debug/faces'+suf+'.dat','r')
faces = []
jf = 0
while jf < nf:
    he = [int(a)-1 for a in f.readline().split()]
    ih = 2*he[0] + he[1]
    wout = make_wire(ih, edges, curves)
    winn = []
    ninner = int(f.readline())
    for i in range(ninner):
        he = [int(a)-1 for a in f.readline().split()]
        ih = 2*he[0] + he[1]
        winn.append(make_wire(ih, edges, curves))
    faces.append(Face(outer=wout,
                      inner=winn,
                      index=jf))
    jf += 1
f.close()


face = faces[iface-1]








#################################################
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1,2)

polys = face.get_polygon()
for i, poly in enumerate(polys):
    if i == 0:
        color = (1,1,0)
    else:
        color = (1,1,1)
    axes[0].add_patch(mpatches.Polygon(xy=poly, closed=True, fc=color, ec='k'))

##########################
if True:
    filebpts = pthin+'brepmesh'+suf2+'/complementary/bpts_'+strf+'.dat'
    filebedg = pthin+'brepmesh'+suf2+'/complementary/bedg_'+strf+'.dat'

    fip = open(filebpts, 'w')
    fie = open(filebedg, 'w')
    #bpoints = []
    #bedges = []
    for f in get_complementary(face):
        polys = f.get_polygon()
        for i, poly in enumerate(polys):
            n = len(poly)
            for j in range(n):
                fip.write('%15s %15s\n' % (poly[j][0], poly[j][1]))
                fie.write('%d %d\n' % (j + 1, (j+1)%n + 1))
                #bpoints.append(poly[j])
                #bedges.append([j, (j+1)%n])
            if i == 0:
                color = (0,1,1)
            else:
                color = (1,1,1)
            axes[1].add_patch(mpatches.Polygon(xy=poly, closed=True, fc=color, ec='k'))
    fip.close()
    fie.close() 
    
    tri, uv, xyz = run_meshgen(filecoefs=pthin+'brepmesh'+suf2+'/c_'+strf+'.cheb',
                               filepoints=filebpts,
                               fileedges=filebedg,
                               fileinfo=pthin+'brepmesh'+suf2+'/info.dat',
                               filetri=pthin+'brepmesh'+suf2+'/complementary/tri_'+strf+'.dat',
                               fileuv=pthin+'brepmesh'+suf2+'/complementary/uv_'+strf+'.dat',
                               filexyz=pthin+'brepmesh'+suf2+'/complementary/xyz_'+strf+'.dat')
    axes[1].triplot(uv[:,0], uv[:,1], tri, '-k')

    tri = numpy.loadtxt(pthin+'brepmesh'+suf2+'/tri_'+strf+'.dat', dtype=int) - 1
    uv = numpy.loadtxt(pthin+'brepmesh'+suf2+'/uv_'+strf+'.dat', dtype=float)
    axes[1].triplot(uv[:,0], uv[:,1], tri, '-b')
    
for ax in axes:
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_aspect('equal')
plt.show()


