import numpy as np
from numpy.polynomial.chebyshev import *
from numpy.fft import fft, ifft

##########################################
def cgl_nodes(N):
    return np.cos(np.arange(0,N+1)*np.pi/float(N))
##########################################
def fcht(f):
    shp = f.shape
    if len( shp ) == 1:
        N = len(f) - 1
        v = np.hstack((f, f[N-1:0:-1]))
        F = fft(v)
        F = F.real / float(N)
        c = F[0:N+1]
        c[0] *= 0.5
        c[N] *= 0.5
    else:
          N = shp[1] - 1
          v = np.hstack((f, f[:,N-1:0:-1]))
          F = fft(v)
          F = F.real / float(N)
          c = F[:,0:N+1]
          c[:,0] = 0.5*c[:,0]
          c[:,N] = 0.5*c[:,N]  
    return c
##########################################
def diff( c ):
    shp = c.shape
    degr = shp[0]-1
    if len( shp ) == 1:
        d = np.zeros( degr+1 )
    else:
        d = np.zeros( (degr+1, shp[1]) ) 
    
    if degr < 1:
        return d
    d[degr-1] = 2.0 * float(degr) * c[degr]
    
    if degr > 1:
        d[degr-2] = 2.0 * float(degr-1) * c[degr-1]
    
    if degr > 2:
        for i in range( degr-3, -1, -1 ):
            d[i] = d[i+2] + 2.0 * float( i+1 ) * c[i+1]
    d[0] *= 0.5
    return d
##########################################
def diff2( c )	:
    shp = c.shape
    degr = [ shp[0]-1, shp[1]-1 ]
    if len( shp ) == 2:
        du = np.zeros( (degr[0]+1,degr[1]+1) )
        dv = np.zeros( (degr[0]+1,degr[1]+1) )
	
        for j in range(degr[1]+1):
            du[:,j] = diff( c[:,j] )
        for i in range(degr[0]+1):
            dv[i,:] = diff( c[i,:] )
    else:
        du = np.zeros( (degr[0]+1,degr[1]+1,shp[2]) )
        dv = np.zeros( (degr[0]+1,degr[1]+1,shp[2]) )
        for j in range(degr[1]+1):
            du[:,j,:] = diff( c[:,j,:] )
        for i in range(degr[0]+1):
            dv[i,:,:] = diff( c[i,:,:] )
    return du, dv
##########################################

##########################################
def nchoosek(n, k):
    if k == 0 or k == n:
        return 1
    else:
        return nchoosek(n-1, k) + nchoosek(n-1, k-1)
##########################################
def factorial(n):
    if n < 2:
        return 1
    else:
        return n*factorial(n-1)
##########################################
def factorial2(n):
    if n < 2:
        return 1
    else:
        return n*factorial2(n-2)
##########################################
def C2Bmatrix( N ):
    A = np.zeros((N,N))
    n = N-1
    for k in range(N):
        for j in range(N):
            for i in range( max(0,j+k-n), min(j,k)+1 ):
                A[j,k] += np.power(-1.0,k-i) * float( nchoosek(2*k, 2*i)*nchoosek(n-k, j-i) )
            A[j,k] /= float( nchoosek(n,j) )
    return A


def B2Cmatrix( N ):
    A = np.zeros((N,N))
    n = N-1
    for k in range(N):
        for j in range(N):
            if j == 0:
                dj0 = 1.0
            else:
                dj0 = 0.0
            for i in range(j+1):
                A[j,k] += np.power(-1.0,j-i)*nchoosek(j,i)*float(factorial2(2*(k+i)-1)*factorial2(2*(n+j-k-i)-1))/float(factorial2(2*i-1)*factorial2(2*(j-i)-1))
            A[j,k] *= float((2-dj0)*factorial2(2*j-1)*nchoosek(n,k))/float(2**(n+j)*factorial(n+j))
    return A
##########################################
def chebfit(x, y, M):
    n = len(x)
    F = np.zeros((M,n))
    F[0,:] = 1.0
    F[1,:] = x
    for i in range(2,M):
        F[i,:] = 2.0*x*F[i-1,:] - F[i-2,:]
    A = np.zeros((M,M))
    b = np.zeros((M,1))
    for j in range(M):
        for i in range(M):
            A[i,j] = np.dot(F[i,:], F[j,:])
    if len(y.shape) == 1:
        b = np.zeros(M)
        for i in range(M):
            b[i] = np.dot(F[i,:],y)
    else:
        b = np.zeros((M,y.shape[1]))
        for j in range(y.shape[1]):
            for i in range(M):
                b[i,j] = np.dot(F[i,:],y[:,j])   
    #A = mymatmul( F, np.transpose(F) )
    #b = mymatmul( F, y )
    c = np.linalg.solve(A, b)
    print('cond(A) = ',np.linalg.cond(A))
    return c
##########################################
def bivariate_to_univariate(c, ivar, ival):
    dim = c.shape[2]
    a = np.zeros((c.shape[(1+ivar)%2], dim))
    if ival == 1:
        for i in range(dim):
            a[:,i] = np.sum(c[:,:,i], axis=ivar)
    else:
        if ivar == 0:
            for i in range(c.shape[1]):
                for j in range(c.shape[0]):
                    a[i] = a[i] + c[j,i]*(-1)**j
        else:
             for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    a[i] = a[i] + c[i,j]*(-1)**j       
    return a
##########################################
def chgvar1(c, x0, x1):
    a = 0.5*(x1 - x0)
    twob = x1 + x0
    b = 0.5*twob

    col = np.zeros((3,c.shape[0]))

    s = np.zeros(c.shape)
    if c.shape[0] == 1:
        s[0] = c[0]
        return s
    col[2,0] = 1.0
    col[1,0] = b
    col[1,1] = a
    s[0] = c[0] + b*c[1]
    s[1] = a*c[1]

    for n in range(2,c.shape[0]):
        col[0,0:n] = twob*col[1,0:n]
        col[0,0:n-1] = col[0,0:n-1] - col[2,0:n-1]
        col[0,1] = col[0,1] + a*col[1,0]
        col[0,1:n+1] = col[0,1:n+1] + a*col[1,0:n]
        col[0,0:n-1] = col[0,0:n-1] + a*col[1,1:n]

        if len(c.shape) < 2:
            s[0:n+1] = s[0:n+1] + c[n]*col[0,0:n+1]
        else:
            for j in range(c.shape[1]):
                s[0:n+1,j] = s[0:n+1,j] + c[n,j]*col[0,0:n+1]

        if n < c.shape[0]-1:
            col[[1,2]] = col[[0,1]]
    return s
##########################################
def chgvar2(c, xy0, xy1):
    s = np.zeros(c.shape)
    if len(c.shape) < 3:
        s = chgvar1(c, xy0[0], xy1[0])
        s = chgvar1(c.T, xy0[1], xy1[1]).T
    else:
        for k in range(c.shape[2]):
            s[:,:,k] = chgvar1(c[:,:,k], xy0[0], xy1[0])
            s[:,:,k] = chgvar1(s[:,:,k].T, xy0[1], xy1[1]).T
    return s
##########################################
def obb_chebyshev1(c):
    # center
    center = c[0]

    # axes
    axes = np.zeros((3,3))
    mag = np.sqrt(np.sum(c[1]**2))
    axes[0] = c[1]/mag

    i = np.argmin(np.absolute(axes[0]))
    axes[1,i] = 1.0
    axes[1] = axes[1] - np.dot(axes[1], axes[0])*axes[0]
    axes[1] = axes[1]/np.sqrt(np.sum(axes[1]**2))
    axes[2] = np.cross(axes[0], axes[1])

    # ranges
    ranges = []
    for dim in range(3):
        r = 0.0
        for m in range(1,c.shape[0]):
            r += abs(np.dot(axes[dim], c[m]))
        ranges.append(r)
    
    return center, ranges, axes.T
##########################################
def obb_chebyshev2(c):   
    # center
    center = c[0,0]

    # axes
    axes = np.zeros((3,3))
    vec = [c[1,0], c[0,1]]
    mag = np.array([np.sum(vec[0]**2), np.sum(vec[1]**2)])
    i = np.argmax(mag)
    mag = np.sqrt(mag)
    axes[0] = vec[i]/mag[i]
    i = (i+1)%2

    axes[2] = np.cross(axes[0], vec[i])
    mag = np.sqrt(np.sum(axes[2]**2))
    if mag < 1.e-8:
        i = np.argmin(np.absolute(axes[0]))
        axes[1,i] = 1.0
        axes[1] = axes[1] - np.dot(axes[1], axes[0])*axes[0]
        axes[1] = axes[1]/np.sqrt(np.sum(axes[1]**2))
        axes[2] = np.cross(axes[0], axes[1])
    else:
        axes[2] = axes[2]/mag
        axes[1] = np.cross(axes[2], axes[0])

    # ranges
    ranges = []
    for dim in range(3):
        r = -abs(np.dot(axes[dim], c[0,0]))
        for m in range(c.shape[0]):
            for n in range(c.shape[1]):
                r += abs(np.dot(axes[dim], c[m,n]))
        ranges.append(r)
    
    return center, ranges, axes.T
##########################################
def read_polynomial2(filename):
    f = open(filename, 'r')
    m, n, d = [int(a) for a in f.readline().split()]
    c = np.zeros((m,n,d))
    for k in range(d):
        for j in range(n):
            for i in range(m):
                c[i,j,k] = float(f.readline())
    return c
