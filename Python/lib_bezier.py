import numpy

def B(coorArr, i, j, t):
    if j == 0:
        return coorArr[i]
    return B(coorArr, i, j - 1, t) * (1 - t) + B(coorArr, i + 1, j - 1, t) * t



def diff(b):
    shp = list(b.shape)
    degr = shp[0]-1
    shp[0] = max(1, degr)
    db = numpy.zeros(tuple(shp))
    for i in range(degr):
        db[i] = b[i+1] - b[i]
    return float(degr)*db

def de_casteljau(b, t):
    bjm1 = numpy.copy(b)
    bj = numpy.zeros(b.shape)
    d = b.shape[0]-1
    n = d+1

    if n == 1:
        return b[0], b[0], b[0]

    bl = numpy.zeros(b.shape)
    br = numpy.zeros(b.shape)

    bl[0] = bjm1[0]
    br[d] = bjm1[d]

    for j in range(1,n):
        for i in range(0,n-j):
            bj[i] = bjm1[i]*(1.0 - t) + bjm1[i+1]*t
        bl[j] = bj[0]
        br[d-j] = bj[d-j]
        bjm1 = numpy.copy(bj)
    f = bj[0]
    return f, bl, br
    


def eval_bezier_curve(b, t):
    dim = b.shape[1]
    if isinstance(t, float) or isinstance(t, int):
        p, bl, br = de_casteljau(b, t)
    else:
        n = len(t)
        p = numpy.zeros((n,dim))
        for i in range(n):
            f, bl, br = de_casteljau(b, t[i])
            for j in range(dim):
                p[i,j] = f[j]
    return p


def AABB_2d_bezier_curve(b):
    return numpy.array([numpy.amin(b, axis=0), numpy.amax(b, axis=0)])


def OBB_2d_bezier_curve(b):
    import sys
    sys.path.append('/d/bandrieu/GitHub/Code/Python/')
    from lib_compgeom import minimal_OBB
    return minimal_OBB(b)
    

def reparameterize_bezier_curve(b, start=0, end=1):
    if start > 0:
        f, bl, b = de_casteljau(b, start)
        end = (end - start)/(1 - start)
    if end < 1:
        f, b, br = de_casteljau(b, end)
    return b
