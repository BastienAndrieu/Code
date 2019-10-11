ROOT = '/d/bandrieu/'#'/home/bastien/'#

import numpy
from numpy.polynomial.chebyshev import chebval, chebval2d

import sys
sys.path.append(ROOT + 'GitHub/Code/Python/')
import lib_chebyshev as lch

class BREP_t:
    def __init__(self, patches=None):
        if patches is None:
            self.patches = []
        else:
            self.patches = patches
        return
    #
    def make_G1_adjacency_matrix(self):
        np = len(self.patches)
        self.G1adjmat = numpy.zeros((np,np), dtype=bool)
        for i in range(np):
            for j in self.patches[i].adj:
                self.G1adjmat[i][j] = True
                self.G1adjmat[j][i] = True
                if i not in self.patches[j].adj: self.patches[j].adj.append(i)
        return
    #
    def trace_tangential_intersection_curves(self, iplanes, hmin, hmax, tolchord):
        np = len(self.patches)
        self.curves = []
        for i, Pi in enumerate(self.patches):
            for j in Pi.adj:
                if i < j:
                    Pj = self.patches[j]
                    if i in iplanes:
                        self.curves.append(
                            trace_plane_intersection_curve(
                                [Pi, Pj],
                                hmin, hmax, tolchord
                            )
                        )
                    elif j in iplanes:
                        self.curves.append(
                            trace_plane_intersection_curve(
                                [Pj, Pi],
                                hmin, hmax, tolchord
                            )
                        )
                    else:
                        self.curves.append(
                            trace_intersection_curve(
                                [Pi, Pj],
                                hmin, hmax, tolchord
                            )
                        )
        return 

        

class Patch_t:
    def __init__(self, xyz, adj=None, tag=None):
        self.xyz = xyz.copy()
        
        if adj is None:
            self.adj = []
        else:
            self.adj = adj

        self.tag = tag
        return

    def get_corners(self):
        return [
            self.xyz[:,0,0],
            self.xyz[:,-1,0],
            self.xyz[:,-1,-1],
            self.xyz[:,0,-1]
        ]

    def get_border(self, border):
        if border == 0:
            return self.xyz[:,:,0]
        elif border == 1:
            return self.xyz[:,-1,:]
        elif border == 2:
            return self.xyz[:,::-1,-1]
        else:
            return self.xyz[:,0,::-1]


class Curve_t:
    def __init__(self, patches, xyz, uv):
        self.patches = patches
        self.xyz = xyz
        self.uv = uv
        return











# CONSTANTS #################
Id3 = numpy.eye(3)

ic2ivar = [1,0,1,0]
ic2ival = [2,1,1,2]

EPS = 1e-9
EPSsqr = EPS**2

EPSuv = 1e-12
EPSxyz = 1e-9
#############################




def rotation(x, y, z, R):
    xr = R[0][0]*x + R[0][1]*y + R[0][2]*z
    yr = R[1][0]*x + R[1][1]*y + R[1][2]*z
    zr = R[2][0]*x + R[2][1]*y + R[2][2]*z
    return xr, yr, zr
    
def translation(x, y, z, t):
    return x + t[0], y + t[1], z + t[2]

def scale(x, y, z, s):
    return s[0]*x, s[1]*y, s[2]*z

def affine_transform(x, y, z, R=Id3, t=[0,0,0], s=[1,1,1]):
    x, y, z = scale(x, y, z, s)
    x, y, z = rotation(x, y, z, R)
    x, y, z = translation(x, y, z, t)
    return x, y, z

def pack_xyz(x, y, z):
    return numpy.array([x,y,z])#numpy.dstack((x, y, z))

def cgl_grid(n, a=1, b=-1):
    x = lch.cgl_nodes(n-1)
    return 0.5*((a-b)*x + a + b)

def matrix_from_columns(cols):
    return numpy.asarray(cols).T

def rotate_basis(b, num_axe, angle):
    c = numpy.cos(angle)
    s = numpy.sin(angle)
    if num_axe == 0:
        R = numpy.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif num_axe == 1:
        R = numpy.array([[s,0,c],[0,1,0],[c,0,-s]])
    else:
        R = numpy.array([[c,-s,0],[s,c,0],[0,0,1]])
    
    v = [i for i in range(3) if i != num_axe]
    v.append(num_axe)
    br = R.dot(b)
    return br[:,v]









def rectangle_cgl(origin, l1, l2, axes, m, n):
    # CGL grid
    u = cgl_grid(m, 0, l1)
    v = cgl_grid(n, 0, l2)
    # Reference XYZ-rectangle 
    x = numpy.tile(u, (n,1)).T
    y = numpy.tile(v, (m,1))
    z = numpy.zeros((m,n))
    # Affine transform
    x, y, z = affine_transform(x, y, z, R=axes, t=origin)
    return pack_xyz(x, y, z)


def cone_cgl(center, r1, r2, l, axes, anglea, angleb, m, n):
    # CGL grid
    t = cgl_grid(m, anglea, angleb)
    r = cgl_grid(n, r1, r2)
    h = cgl_grid(n, 0, l)
    # Reference XYZ-cone
    x = numpy.outer(numpy.cos(t), r)
    y = numpy.outer(numpy.sin(t), r)
    z = numpy.tile(h, (m,1))
    # Affine transform
    x, y, z = affine_transform(x, y, z, R=axes, t=center)
    return pack_xyz(x, y, z)


def cylinder_cgl(center, r, l, axes, anglea, angleb, m, n):
    # CGL grid
    t = cgl_grid(m, anglea, angleb)
    h = cgl_grid(n, 0, l)
    # Reference XYZ-cone
    x = r*numpy.tile(numpy.cos(t), (n,1)).T
    y = r*numpy.tile(numpy.sin(t), (n,1)).T
    z = numpy.tile(h, (m,1))
    # Affine transform
    x, y, z = affine_transform(x, y, z, R=axes, t=center)
    return pack_xyz(x, y, z)


def torus_cgl(center, r1, r2, axes, angle1a, angle1b, angle2a, angle2b, m, n):
    # CGL grid
    t1 = cgl_grid(m, angle1a, angle1b)
    t2 = cgl_grid(m, angle2a, angle2b)
    # Reference XYZ-torus
    r = r1 + r2*numpy.cos(t2)
    x = numpy.outer(numpy.cos(t1), r)
    y = numpy.outer(numpy.sin(t1), r)
    z = r2*numpy.tile(numpy.sin(t2), (m,1))
    # Affine transform
    x, y, z = affine_transform(x, y, z, R=axes, t=center)
    return pack_xyz(x, y, z)

def bilinear_cgl(a, b, c, d, m, n):
    # CGL grid
    u = cgl_grid(m, -1, 1)
    v = cgl_grid(n, -1, 1)
    # Bilinear XYZ-patch
    xyz = numpy.zeros((3,m,n))
    for i in range(3):
        xyz[i] = 0.25*(a[i]*numpy.outer(1-u,1-v) + b[i]*numpy.outer(1+u,1-v) + c[i]*numpy.outer(1+u,1+v) + d[i]*numpy.outer(1-u,1+v))
    return xyz

def symmetry(xyz, iaxe):
    xyz_s = xyz[:,::-1].copy()
    xyz_s[iaxe] = -xyz_s[iaxe]
    return xyz_s


def cht_xyz(xyz):
    c = numpy.zeros((xyz.shape[1], xyz.shape[2], 3))
    for i in range(3):
        c[:,:,i] = lch.fcht(lch.fcht(xyz[i]).T).T
    return c



#def trace_tangential_intersection_curve(self, iplanes, hmin, hmax, tolchord)
def trace_intersection_curve(patches, hmin, hmax, tolchord):
    xc = [Pa.get_corners() for Pa in patches]
    for jc in range(4):
        for ic in range(4):
            v1 = xc[0][ic] - xc[1][(jc+1)%4]
            v2 = xc[0][(ic+1)%4] - xc[1][jc]
            if max(v1.dot(v1), v2.dot(v2)) < EPS**2:
                xyz = patches[1].get_border(jc)
                c = lch.fcht(xyz).T
                xyz, t = discretize_curve(c, hmin, hmax, tolchord, ta=1, tb=-1)
                uv = numpy.zeros((2,2,len(t)))
                jvar = ic2ivar[jc]
                jval = ic2ival[jc]
                uv[1,jvar,:] = (-1)**jval
                uv[1,(jvar+1)%2] = numpy.sign(1.5 - jc)*t
                ivar = ic2ivar[ic]
                ival = ic2ival[ic]
                uv[0,ivar,:] = (-1)**ival
                uv[0,(ivar+1)%2] = -numpy.sign(1.5 - ic)*t
                return Curve_t(patches=patches, xyz=xyz, uv=uv)
    return None
    

def trace_plane_intersection_curve(patches, hmin, hmax, tolchord):
    ori = patches[0].xyz[:,-1,-1]
    nor = numpy.cross(patches[0].xyz[:,0,-1] - ori, patches[0].xyz[:,-1,0] - ori)
    # find correct border 
    fc = numpy.array([nor.dot(xc - ori) for xc in patches[1].get_corners()])
    for ic in range(4):
        if numpy.amax(numpy.absolute(fc[[ic, (ic+1)%4]])) < EPS:
            xyz = patches[1].get_border(ic)
            c = lch.fcht(xyz).T
            xyz, t = discretize_curve(c, hmin, hmax, tolchord, ta=1, tb=-1)
            ivar = ic2ivar[ic]
            ival = ic2ival[ic]
            uv = numpy.zeros((2,2,len(t)))
            uv[1,ivar,:] = (-1)**ival
            uv[1,(ivar+1)%2] = numpy.sign(1.5 - ic)*t
            break
    # project onto other plane surface
    for ip in range(len(t)):
        c = cht_xyz(patches[0].xyz)
        cu, cv = lch.diff2(c)
        u, v = uv[0,:,ip]
        # Newton
        for it in range(20):
            s = chebval2d(u, v, c)
            r = xyz[:,ip] - s
            su = chebval2d(u, v, cu)
            sv = chebval2d(u, v, cv)
            E = su.dot(su)
            F = su.dot(sv)
            G = sv.dot(sv)
            invdet = 1/(E*G - F**2)
            rsu = r.dot(su)
            rsv = r.dot(sv)
            du = (rsu*G - rsv*F)*invdet
            dv = (rsv*E - rsu*F)*invdet
            u += du
            v += dv
            if du**2 + dv**2 < EPSuv**2 and r.dot(r) < EPSxyz**2:
                uv[0,0,ip] = u
                uv[0,1,ip] = v
                break
    #
    return Curve_t(patches=patches, xyz=xyz, uv=uv)

def squared_curvature_curve(xt, yt, zt, xtt, ytt, ztt):
    u = yt*ztt - zt*ytt
    v = zt*xtt - xt*ztt
    w = xt*ytt - yt*xtt
    return (u**2 + v**2 + w**2)/numpy.maximum(EPSsqr, (xt**2 + yt**2 + zt**2)**3)


def discretize_curve(c, hmin, hmax, tolchord, ta=-1, tb=1, n0=20):
    FRACsqr = 4*tolchord*(2 - tolchord)
    c_t = lch.diff(c)
    c_tt = lch.diff(c_t)
    tbounds = [min(ta, tb), max(ta, tb)]
    sdtab = numpy.sign(tb - ta)
    #
    t = numpy.linspace(ta, tb, n0)
    xyz = chebval(t, c)
    xyz_t = chebval(t, c_t)
    xyz_tt = chebval(t, c_tt)
    curvaturesqr = squared_curvature_curve(
        xyz_t[0], xyz_t[1], xyz_t[2],
        xyz_tt[0], xyz_tt[1], xyz_tt[2],
    )

    hminsqr = hmin**2
    hmaxsqr = hmax**2
    hsqr = numpy.minimum(hmaxsqr, numpy.maximum(hminsqr, FRACsqr/curvaturesqr))
    #
    maxit = 100*n0
    for it in range(maxit):
        changes = False
        for i in range(len(t)-1):
            ei = xyz[:,i+1] - xyz[:,i]
            lisqr = ei.dot(ei)
            if lisqr > 2*max(hsqr[i], hsqr[i+1]):
                # split
                hi = numpy.sqrt(hsqr[i])
                hj = numpy.sqrt(hsqr[i+1])
                tm = (hj*t[i] + hi*t[i+1])/(hi + hj)
                xyzm = chebval(tm, c)
                xyzm_t = chebval(tm, c_t)
                xyzm_tt = chebval(tm, c_tt)
                curvaturemsqr = squared_curvature_curve(
                    xyzm_t[0], xyzm_t[1], xyzm_t[2],
                    xyzm_tt[0], xyzm_tt[1], xyzm_tt[2],
                )
                hmsqr = numpy.minimum(hmaxsqr, numpy.maximum(hminsqr, FRACsqr/curvaturemsqr))
                t = numpy.insert(t, i+1, tm)
                xyz = numpy.insert(xyz, i+1, xyzm, axis=1)
                hsqr = numpy.insert(hsqr, i+1, hmsqr)
                changes = True
                break
            elif lisqr < 0.5*min(hsqr[i], hsqr[i+1]):
                if xyz.shape[1] <= 2: break
                # collapse
                if i == 0:
                    # remove i+1
                    t = numpy.delete(t, i+1)
                    xyz = numpy.delete(xyz, i+1, axis=1)
                    hsqr = numpy.delete(hsqr, i+1)
                elif i == xyz.shape[1]-2:
                    # remove i
                    t = numpy.delete(t, i)
                    xyz = numpy.delete(xyz, i, axis=1)
                    hsqr = numpy.delete(hsqr, i)
                else:
                    # relocate i+1
                    hi = numpy.sqrt(hsqr[i])
                    hj = numpy.sqrt(hsqr[i+1])
                    t[i+1] = (hj*t[i] + hi*t[i+1])/(hi + hj)
                    xyz[:,i+1] = chebval(t[i+1], c)
                    xyzm_t = chebval(t[i+1], c_t)
                    xyzm_tt = chebval(t[i+1], c_tt)
                    curvaturemsqr = squared_curvature_curve(
                        xyzm_t[0], xyzm_t[1], xyzm_t[2],
                        xyzm_tt[0], xyzm_tt[1], xyzm_tt[2],
                    )
                    hsqr[i+1] = numpy.minimum(hmaxsqr, numpy.maximum(hminsqr, FRACsqr/curvaturemsqr))
                    # remove i
                    t = numpy.delete(t, i)
                    xyz = numpy.delete(xyz, i, axis=1)
                    hsqr = numpy.delete(hsqr, i)
                changes = True
                break
            else: continue
        if not changes:break
    #
    """
    for i in range(len(t)):
        if i == 0:
            print('%s' % numpy.sqrt(hsqr[i]))
        else:
            print('%s\t%s' % (
                numpy.sqrt(hsqr[i]),
                numpy.sqrt(numpy.dot(xyz[:,i] - xyz[:,i-1], xyz[:,i] - xyz[:,i-1]))
            ))
    """
    #
    return xyz, t
