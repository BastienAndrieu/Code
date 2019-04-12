import math

###################################
def get_dimensions(v):
    m = len(v)
    if isinstance(v[0], (list,)):
        n = len(v[0])
    else:
        n = 1
    return m, n
###################################
def init_vector(m, n=1, val=0.0):
    v = []
    if n > 1:
        for i in range(m):
            row = []
            for j in range(n):
                row.append(val)
            v.append(row)
    else:
        for i in range(m):
            v.append(val)
    return v
###################################
def norm(a):
    n, dim = get_dimensions(a)
    norma = 0.0
    for i in range(n):
        if dim > 1:
            for j in range(dim):
                norma += a[i][j]**2
        else:
            norma += a[i]**2
    return math.sqrt(norma)
###################################
def normdiff(a, b):
    n, dim = get_dimensions(a)
    diff = 0.0
    for i in range(n):
        if dim > 1:
            for j in range(dim):
                diff += (a[i][j] - b[i][j])**2
        else:
            diff += (a[i] - b[i])**2
    return math.sqrt(diff)
###################################
def matvecprod(A, x):
    n, dim = get_dimensions(x)
    b = init_vector(n, dim)
    for k in range(len(A)):
        i   = A[k][0]
        j   = A[k][1]
        aij = A[k][2]
        if dim > 1:
            for l in range(dim):
                b[i][l] += aij*x[j][l]
        else:
            b[i] += aij*x[j]   
    return b
###################################
def solve_jacobi(A, b, tol=1e-7, itmax=100, eps=1e-7):
    n, dim = get_dimensions(b)
    x = init_vector(n, dim)
    xtmp = init_vector(n, dim)
    invaii = init_vector(n)
    for k in range(len(A)):
        i   = A[k][0]
        j   = A[k][1]
        aij = A[k][2]
        if i == j:
            if abs(aij) < eps:
                return True, x
            else:
                invaii[i] = 1.0/aij
    
    for it in range(itmax):
        for i in range(n):
            if dim > 1:
                for l in range(dim):
                    xtmp[i][l] = b[i][l]
            else:
                xtmp[i] = b[i]
        for k in range(len(A)):
            i   = A[k][0]
            j   = A[k][1]
            aij = A[k][2]
            if i != j:
                if dim > 1:
                    for l in range(dim):
                        xtmp[i][l] -= aij*x[j][l]
                else:
                     xtmp[i] -= aij*x[j]
        for i in range(n):
            if dim > 1:
                for l in range(dim):
                    xtmp[i][l] *= invaii[i]
            else:
                xtmp[i] *= invaii[i]
        diff = normdiff(xtmp, x)/norm(xtmp)
        res = normdiff(matvecprod(A, xtmp), b)/norm(b)
        print 'it.#'+str(it+1)+': delta* =',diff,', res* =',res
        x = list(xtmp) # copy
        if diff < tol or res < eps:
            break
    return x
