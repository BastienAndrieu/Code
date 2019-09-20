import numpy
import time
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
from lib_chebyshev import C2Bmatrix, C2Bmatrix_new

def mat_as_str(A):
    s = ''
    for i in range(A.shape[0]):
        s += '| '
        for j in range(A.shape[1]):
            s += '%s ' % A[i,j]
        s += '|\n'
    return s

args = sys.argv
if len(args) > 1:
    n0 = int(args[1])
    if len(args) > 2:
        n1 = int(args[2])
    else:
        n1 = n0
else:
    n0 = 2


    


for n in range(n0,n1+1):
    print '\n\n\n'
    print 'n = %d' % n
    start = time.time()
    A = C2Bmatrix(n)
    tA = time.time() - start

    start = time.time()
    B = C2Bmatrix_new(n)
    tB = time.time() - start
    #print A-B, '\n'
    print 'tA = %s, tB = %s, tA/tB = %s' % (tA, tB, tA/tB)
    if False:#n < 6:
        print 'A = \n' + mat_as_str(A) + '\nB = \n' + mat_as_str(B) + '\n'
    #
    print 'max|A - B| = %s' % numpy.amax(numpy.absolute(A - B))
        
