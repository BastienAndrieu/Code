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
