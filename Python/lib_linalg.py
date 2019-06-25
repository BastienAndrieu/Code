import numpy

# Basic Linear Algebra
EPSlinag = 1.e-7

### solve square linear system using Gaussian elimination ###
def solve_NxN(A, b):
    nrow, ncol = A.shape
    n = min(nrow, ncol)
    C = numpy.hstack([A[0:n,0:n] ,numpy.expand_dims(b[0:n],1)])
    C, singular = gauss_elim(C)
    if singular:
        return numpy.zeros(n), singular
    else:
        return solve_up_tri(C[:,0:n], C[:,n]), False
###########################


### solve an upper triangular linear system ###
def solve_up_tri(U, b):
    nrow, ncol = U.shape
    sol = numpy.zeros(ncol)
    n = min(nrow, ncol)
    sol[n-1] = b[n-1]/U[n-1,n-1]
    for i in range(n-2,-1,-1):
        sol[i] = (b[i] - numpy.sum(sol[i+1:n]*U[i,i+1:n])) / U[i,i]
    return sol
###########################

### perform Gaussian elimination ###
def gauss_elim(A):
    nrow, ncol = A.shape
    singular = False
    n = min(nrow, ncol)
    for k in range(n):
        i_max = k + numpy.argmax(numpy.absolute(A[k:,k])) # pivot
        
        if abs(A[i_max,k]) < EPSlinag: singular = True

        # swap rows i_max / k
        A[[i_max,k]] = A[[k, i_max]]
        
        invAkk = 1./A[k,k]
        for i in range(k+1,nrow):
            A[i,k+1:] = A[i,k+1:] - A[k,k+1]*A[i,k]*invAkk
            A[i,k] = 0.0
    return A, singular
###########################



def matvecprod(A,b):
    m, n = A.shape
    c = numpy.zeros(m)
    for i in range(m):
        for j in range(n):
            c[i] += A[i,j]*b[j]
    return c


def matmul(A,B):
    m, n = A.shape
    n, p = B.shape
    C = numpy.zeros((m,p))
    for i in range(m):
        for j in range(p):
            C[i,j] = numpy.dot(A[i,:], B[:,j])
    return C
