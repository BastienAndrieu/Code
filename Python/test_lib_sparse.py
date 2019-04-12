import sys
import numpy

sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_sparse as lsp

#A = numpy.array([[1.0,2.0],[3.0,4.0]])
#b = [17.0, 39.0]
A = numpy.array([[5.0, -2.0, 3.0],
                 [-3.0, 9.0, 1.0],
                 [2.0, -1.0, -7.0]])
b = [-1.0, 2.0, 3.0]
                  

Asp = []
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Asp.append([i,j,float(A[i,j])])

x = lsp.solve_jacobi(Asp, b, itmax=100)
print '\n'

b = numpy.asarray(b)
x = numpy.linalg.solve(A,b)
print x
print '\n'
