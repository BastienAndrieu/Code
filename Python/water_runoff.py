ROOT = '/d/bandrieu/'#'/home/bastien/'#

import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
sys.path.append(ROOT+'GitHub/Code/Python')
import lib_halfedge as lhe


def fix_uv_triangle(uv):
    u, v = uv
    if u < 0:
        if v < 0:
            return numpy.array([0,0])
        if v > 1:
            return numpy.array([0,1])
        else:
            return numpy.array([0,v])
    if v < 0:
        if u > 1:
            return numpy.array([1,0])
        else:
            return numpy.array([u,0])
    if u + v > 1:
        if v > 1 + u:
            return numpy.array([0,1])
        elif v < -1 + u:
            return numpy.array([1,0])
        else:
            a = v - u
            u = (1 - a)/2
            v = u + a
            return numpy.array([u,v])
    return uv
    

"""
fig, ax = plt.subplots()
ax.plot([0,1,0,0], [0,0,1,0], '0.2')
for i in range(20):
    uv = 2*numpy.random.rand(2)-1
    x = numpy.vstack([uv, fix_uv_triangle(uv)])
    ax.plot(x[:,0], x[:,1], 'r.-')
ax.set_aspect('equal')
plt.show()
exit()
"""










    
def random_uv_in_triangle():
    uv = numpy.random.rand(2)
    return numpy.random.rand()*uv/numpy.sum(uv)


def uv_vertex(ivert):
    if ivert == 0:
        uv = [0,0]
    elif ivert == 1:
        uv = [1,0]
    else:
        uv = [0,1]
    return numpy.array(uv)

    
def veclensqr(vec):
    return vec.dot(vec)

def veclen(vec):
    return numpy.sqrt(veclensqr(vec))

def baryco_to_co(mesh, iface, baryco, normalize=True):
    m = min(len(baryco), len(mesh.f2v[iface]))
    baryco = numpy.asarray(baryco[:m])
    if normalize: baryco = baryco/numpy.sum(baryco)
    #
    dim = 3
    co = numpy.zeros(3)
    for i in range(m):
        v = mesh.verts[mesh.f2v[iface][i]]
        dimv = len(v.co)
        dim = min(dim, dimv)
        co[:dimv] = co[:dimv] + baryco[i]*v.co
    return co[:dim]

def face_normal(mesh, iface, normalize=True):
    verts = mesh.f2v[iface]
    xyz = [mesh.verts[i].co for i in verts]
    m = len(verts)
    if m == 3:
        nor = triangle_normal(xyz)
    else:
        nor = numpy.zeros(3)
        for i in range(m):
            j = (i+1)%m
            k = (i+2)%m
            nor = nor + triangle_normal([xyz[i], xyz[j], xyz[k]], False)
    if normalize:
        norlen = veclen(nor)
        if norlen > 1e-10:
            nor = nor/norlen
    return nor


def triangle_normal(xyz):
    u = xyz[1] - xyz[0]
    v = xyz[2] - xyz[0]
    return numpy.cross(u, v)


def xyz_to_uv(mesh, iface, xyz):
    verts = [mesh.verts[i] for i in mesh.f2v[iface]]
    a = verts[1].co - verts[0].co
    b = verts[2].co - verts[0].co
    M = [a.dot(a), a.dot(b), b.dot(b)]
    det = M[0]*M[2] - M[1]*M[1]
    if abs(det) < 1e-12:
        return None
    else:
        invdet = 1/det
        r = a.dot(xyz - verts[0].co)
        s = b.dot(xyz - verts[0].co)
        u = (r*M[2] - s*M[1])*invdet
        v = (s*M[0] - r*M[1])*invdet
        return numpy.array([u,v])
            


def uv_to_xyz(mesh, iface, uv):
    verts = [mesh.verts[i] for i in mesh.f2v[iface]]
    a = verts[1].co - verts[0].co
    b = verts[2].co - verts[0].co
    return verts[0].co[:] + uv[0]*a[:] + uv[1]*b[:]



def check_uv_inside(uv):
    EPS = 1e-12
    for i in range(2):
        if uv[i] < -EPS or uv[i] > 1+EPS: return False
    if uv[0] + uv[1] > 1+EPS: return False
    return True



def direction_field(xyz):
    if True:
        return gravity[:]
    else:
        x, y, z = xyz
        r = numpy.hypot(x, y)
        x /= r
        y /= r
        h = 0.1
        f = numpy.sqrt(1 - h**2)
        return numpy.array([f*y, -f*x, h])




def runoff_step(uv_start, iface_start, mesh, step_size, EPSflow=1e-6):
    VERBOSE = False
    #
    if VERBOSE:
        print '\trunoffstep:'
        print '\t\tuv, iface:', uv_start, iface_start
    #
    in_mesh = True
    uv = uv_start[:]
    iface = iface_start
    # get xyz coords of start point
    xyz = uv_to_xyz(mesh, iface, uv)
    if VERBOSE: print '\t\txyz = ', xyz
    # get flow direction
    flow = direction_field(xyz)
    # project onto local tangent plane
    uv_target = xyz_to_uv(mesh, iface, xyz + flow)
    duv = uv_target - uv
    verts = [mesh.verts[i] for i in mesh.f2v[iface]]
    a = verts[1].co - verts[0].co
    b = verts[2].co - verts[0].co
    flow = duv[0]*a + duv[1]*b
    flowlen = veclen(flow)
    if flowlen < EPSflow:
        uv_target = None
        jface = None
        return uv_target, jface, in_mesh, flow
    # rescale to step size
    flow = flow*step_size/flowlen
    if VERBOSE: print '\t\tflow = ', flow
    # get flow in uv-space
    uv_target = xyz_to_uv(mesh, iface, xyz + flow)
    duv = uv_target - uv
    if VERBOSE:
        print '\t\tuv_target = ', uv_target
        print '\t\tcap =', numpy.degrees(numpy.arctan2(duv[1], duv[0]))
    # check edge crossings
    crossed_edge, lamb = check_edge_crossing_uv(uv, duv)
    if crossed_edge is None or lamb is None:
        return uv_target, iface, in_mesh, flow
    else:
        twin = mesh.get_twin([iface, crossed_edge])
        if twin is None:
            in_mesh = False
            jface = iface
        else:
            jface = twin[0]
            crossed_edge = twin[1]
            lamb = 1 - lamb
        #
        uv_cross = (1 - lamb)*uv_vertex(i) + lamb*uv_vertex((i+1)%3)
        if crossed_edge == 0:
            uv_target[0] = lamb
            uv_target[1] = 0
        if crossed_edge == 1:
            uv_target[0] = 1 - lamb
            uv_target[1] = lamb
        if crossed_edge == 2:
            uv_target[0] = 0
            uv_target[1] = 1 - lamb
        return uv_target, jface, in_mesh, flow
    

def check_edge_crossing_uv(uv0, duv):
    VERBOSE = False
    u0, v0 = uv0
    du, dv = duv
    abc = [[0,1,0], [1,1,1], [1,0,0]]
    tmax = -1
    lamb = None
    crossed_edge = None
    if check_uv_inside(uv0 + duv):
        return crossed_edge, lamb
    else:
        for i in range(3):
            if VERBOSE: print '\t\t\tedge #', i
            a, b, c = abc[i]
            numer = c - a*u0 - b*v0
            denom = a*du + b*dv
            if abs(denom) < 1e-12:
                if abs(numer) < 1e-12:
                    t = 0
                    if VERBOSE: print '\t\t\tt = ', t
                    if t > tmax:
                        tmax = t
                        crossed_edge = i
                        uv_cross = [u0, v0]
            else:
                t = numer/denom
                if VERBOSE:
                    print '\t\t\tt = ', t
                    print '\t\t\tcross at uv = ', u0 + t*du, v0 + t*dv, ' sum =', u0 + t*du + v0 + t*dv
                if t > -1e-12 and t < 1+1e-12 and t > tmax:
                    if check_uv_inside([u0 + t*du, v0 + t*dv]):
                        tmax = t
                        crossed_edge = i
                        uv_cross = [u0 + t*du, v0 + t*dv]
    #
    if crossed_edge == 0:
        lamb = uv_cross[0]
    elif crossed_edge == 1:
        lamb = 0.5*(1 + uv_cross[1] - uv_cross[0])
    elif crossed_edge == 2:
        lamb = 1 - uv_cross[1]
    return crossed_edge, lamb
#########################################




#########################################
# IMPORT TERRAIN MESH
print 'read terrain mesh...'
pref = 'Blender/terrain_'
xyz = numpy.loadtxt(pref + 'xyz.dat', dtype=float)
f2v = numpy.loadtxt(pref + 'f2v.dat', dtype=int)
all_tri = all([len(f) == 3 for f in f2v])

if not all_tri:
    # triangulate...
    print 'must triangulate...'
    exit()

aabb = [numpy.amin(xyz, axis=0), numpy.amax(xyz, axis=0)]
range_xyz = aabb[1] - aabb[0]
mesh_size = veclen(range_xyz)

print '   ok.'
#########################################

#########################################
# MAKE HALFEDGE DS
print 'make halfedge DS...'
mesh = lhe.pydata_to_SurfaceMesh(xyz, f2v)
print '   ok.'
#########################################

#########################################
# WATER RUNOFF PARAMETERS
#gravity = numpy.array([0,0,-1])
gravity = 2*numpy.random.rand(3)
gravity = gravity/veclen(gravity)

n_sources = 1#20
step_size = mesh_size*1e-2

n_steps_max = 10000
length_max = 10*mesh_size

EPS = 1e-6
#########################################

#########################################
# WATER RUNOFF SIMULATION
runoffs = []
flowss = []
for isource in range(n_sources):
    print '\tsource #%d/%d' % (isource+1, n_sources)
    iface = numpy.random.randint(len(mesh.f2v))
    uv = random_uv_in_triangle()
    xyz = uv_to_xyz(mesh, iface, uv)
    #
    runoff = [[xyz, iface]]
    flows = []
    length = 0
    for step in range(n_steps_max):
        print '\t\tstep #%d' % step
        uv_new, iface_new, in_mesh, flow = runoff_step(
            uv,
            iface,
            mesh,
            step_size,
            EPSflow=1e-6
        )
        flows.append(flow)
        #print '\t-->', uv_new, iface_new
        if uv_new is None or iface_new is None: break
        #uv_new = fix_uv_triangle(uv_new)
        if not check_uv_inside(uv_new):
            print 'INVALID UV: ', uv_new
            break
        xyz_new = uv_to_xyz(mesh, iface_new, uv_new)
        dlength = veclen(xyz_new - runoff[-1][0])
        length += dlength
        #print '\n\t\t\txyznew =', xyz_new
        uv = uv_new[:]
        iface = iface_new
        runoff.append([xyz_new, iface_new])
        if not in_mesh:
            print 'REACHED MESH BOUNDARY'
            break
        if length > length_max:
            print 'REACHED MAX LENGTH'
            break
        if dlength < step_size*1e-3:
            print 'STATIONNARY (dlength/step_size = %s)' % (dlength/step_size)
            break
    #
    #if veclen(runoff[0][0] - runoff[-1][0]) > 10*step_size:
    runoffs.append(runoff)
    flowss.append(flows)
#########################################













#########################################


f = open(ROOT+'GitHub/Code/Python/Blender/runoffs.dat', 'w')
f.write('%d\n' % len(runoffs))
for runoff in runoffs:
    f.write('%d\n' % len(runoff))
    for xyz, iface in runoff:
        x, y, z = xyz
        f.write('%s %s %s %d\n' % (x, y, z, iface))
f.close()




#########################################
# VISU (2d)
fig, ax = plt.subplots()
xm = [v.co[0] for v in mesh.verts]
ym = [v.co[1] for v in mesh.verts]
zm = [v.co[2] for v in mesh.verts]
tri = [f[0:3] for f in mesh.f2v]

ax.tripcolor(xm, ym, tri, zm, cmap=cm.terrain, shading='gouraud')
#mesh.plot_as_triangulation(ax, color='0.3', linewidth=0.5)


for i, runoff in enumerate(runoffs):
    xyz = [r[0] for r in runoff]
    x = [p[0] for p in xyz]
    y = [p[1] for p in xyz]
    z = [p[2] for p in xyz]
    ax.plot(x, y, 'k.-', lw=1.2)
    ax.plot(x[0], y[0], 'k*', ms=10)
    #
    for j in range(len(runoff)-1):
        ax.arrow(
            runoff[j][0][0], runoff[j][0][1], flowss[i][j][0], flowss[i][j][1],
            color='r',
            width=5e-5,
            head_width=3e-4,
            length_includes_head=True
        )

ax.set_aspect('equal')
plt.show()
#########################################
