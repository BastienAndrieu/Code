ROOT = '/d/bandrieu/'#'/home/bastien/'

import numpy
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

class Patch_t:
    def __init__(self, xyz, adj=None, tag=None):
        self.xyz = xyz.copy()
        
        if adj is None:
            self.adj = []
        else:
            self.adj = adj

        self.tag = tag
        return
















Id3 = numpy.eye(3)

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
