import numpy
EPSlp = 2.22e-16
BIGlp = 1.e6

def lp_solve(x, A, c):
    stat = 0
    dim = len(c)
    print 'dim = %d' % dim
    #
    if dim == 1:
        return lp_solve_1d(x[0], A, c[0])
    #
    for i in range(dim,A.shape[0]):
        if A[i,:dim].dot(x) + A[i,dim] > (dim+1)*EPSlp: continue
        # the provisional optimal point x is on the wrong side of the hyperplane 
        # associated to the i-th constraint
        #
        # find the largest coefficient of that hyperplane's equation
        l = numpy.argmax(numpy.absolute(A[i,:dim]))
        #
        if abs(A[i,l]) < EPSlp: continue
        #
        j = [k for k in range(dim) if k != l]
        #
        # project constraints up to i-1 and the objective function to lower dimension
        inv_Ail = 1.0/A[i,l]
        Ai = numpy.zeros((i,dim))
        for k in range(i):
            Ai[k,:-1] = A[k,j] - A[k,l]*A[i,j]*inv_Ail
            Ai[k,-1] = A[k,dim] - A[k,l]*A[i,dim]*inv_Ail
        ci = c[j] - c[l]*A[i,j]*inv_Ail

        # solve lower-dimensional LP subproblem
        xj = x[j]
        stat, xj = lp_solve(xj, Ai, ci)

        if stat > 0: break

        # back substitution
        x[j] = xj
        x[l] = -(A[i,dim] + A[i,j].dot(xj))*inv_Ail
    return stat, x
        
def lp_solve_1d(x, A, c):
    stat = 0
    L = -BIGlp
    R =  BIGlp
    #
    n = A.shape[0]
    np = 0
    #
    for i in range(n):
        if A[i,0] < -EPSlp:
            np += 1
            R = min(R, -A[i,1]/A[i,0])
        elif A[i,0] > EPSlp:
            L = max(L, -A[i,1]/A[i,0])
    #
    if np == n:
        x = L
        if c < -EPSlp: stat = -1
    elif np == 0:
        x = R
        if c > EPSlp: stat = -1
    else:
        if L > R + EPSlp:
            stat = 1
        else:
            if c < -EPSlp:
                x = R
            elif c > EPSlp:
                x = L
            else:
                if abs(L) > abs(R) + EPSlp:
                    x = R
                else:
                    x = L
    #
    return stat, x
                    
