import numpy

def B(coorArr, i, j, t):
    if j == 0:
        return coorArr[i]
    return B(coorArr, i, j - 1, t) * (1 - t) + B(coorArr, i + 1, j - 1, t) * t











def de_casteljau(b, t):
    bjm1 = numpy.copy(b)
    bj = numpy.zeros(b.shape)
    d = b.shape[0]-1
    n = d+1

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
    
