import numpy
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')

import lib_linalg

args = sys.argv
if len(args) < 2:
    nrow = 2
else:
    nrow = int(args[1])
if len(args) < 3:
    ncol = 2
else:
    ncol = int(args[2])

A = 2.*numpy.random.rand(nrow,ncol) - 1.
b = 2.*numpy.random.rand(nrow) - 1.

sol, sing = lib_linalg.solve_NxN(A,b)

res = lib_linalg.matvecprod(A, sol)

print numpy.absolute(res - b)
